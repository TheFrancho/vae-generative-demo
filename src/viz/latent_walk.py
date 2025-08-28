import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.cvae import CVAE
from src.data.dataset import make_loaders


def _to_one_hot(y_idx, num_classes, device):
    y = torch.zeros((y_idx.shape[0], num_classes), device=device)
    y[torch.arange(y_idx.shape[0]), y_idx] = 1.0
    return y


def _decode_grid(model, y_oh, zmin, zmax, steps, out_png):
    zs = []
    lin = np.linspace(zmin, zmax, steps)

    for i in range(steps):
        for j in range(steps):
            zs.append([lin[i], lin[j]])
    z = torch.tensor(zs, dtype=torch.float32, device=y_oh.device)
    y_rep = y_oh.repeat(steps*steps, 1)

    with torch.no_grad():
        x_hat = model.decode(z, y_rep).cpu().numpy()

    imgs = x_hat[:,0]
    H,W = imgs.shape[-2], imgs.shape[-1]
    canvas = np.zeros((steps*H, steps*W), dtype=np.float32)
    k = 0

    for i in range(steps):
        for j in range(steps):
            canvas[i*H:(i+1)*H, j*W:(j+1)*W] = imgs[k]; k+=1

    plt.figure(figsize=(6,6))
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=1); plt.axis("off")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved latent grid -> {out_png}")


def _encode_class_centroid(model, loader, class_id, device):
    mus = []
    with torch.no_grad():
        for x, y, y_oh, _ in loader:
            x = x.to(device); y = y.to(device); y_oh = y_oh.to(device)
            mask = (y == class_id)
            if mask.any():
                x_sel = x[mask]; y_oh_sel = y_oh[mask]
                mu, _ = model.encode(x_sel, y_oh_sel)
                mus.append(mu.cpu())
    if not mus:
        raise ValueError(f"No samples for class {class_id}.")
    return torch.cat(mus, 0).mean(0, keepdim=True).to(device)


def _interp_pair_frames(model, y_a, y_b, z_a, z_b, steps):
    frames = []
    with torch.no_grad():
        for t in np.linspace(0, 1, steps):
            z_t = (1-t)*z_a + t*z_b
            y_t = (1-t)*y_a + t*y_b  # soft label blend
            x_hat = model.decode(z_t, y_t).cpu().numpy()[0,0]
            frames.append((x_hat*255).astype(np.uint8))
    return frames


def _save_gif(frames, out_gif, fps=10):
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames, duration=1.0/fps)
    print(f"Saved GIF -> {out_gif}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default="outputs/datasets/shapes_10cls.npz")
    ap.add_argument("--mode", type=str, choices=["grid","interp","sequence","randomseq"], required=True)
    ap.add_argument("--class-for-grid", type=int, default=None)
    ap.add_argument("--z-range", type=float, nargs=2, default=[-3.0, 3.0])
    ap.add_argument("--steps", type=int, default=20, help="steps per interpolation")
    ap.add_argument("--outdir", type=str, default="outputs/figures/latent")
    ap.add_argument("--class-a", type=int, default=0)
    ap.add_argument("--class-b", type=int, default=-1)
    ap.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    ap.add_argument("--z-dim", type=int, default=2)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader = make_loaders(args.data, batch_size=256, one_hot_for_cvae=True)
    loader = {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}[args.split]
    num_classes = loader.dataset.num_classes

    model = CVAE(num_classes=num_classes, z_dim=args.z_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    if args.mode == "grid":
        if args.z_dim != 2: raise ValueError("Grid mode requires z_dim=2.")
        c = args.class_for_grid if args.class_for_grid is not None else (num_classes-1)
        y_oh = _to_one_hot(torch.tensor([c], dtype=torch.long, device=device), num_classes, device)
        _decode_grid(model, y_oh, args.z_range[0], args.z_range[1], args.steps, f"{args.outdir}/grid_class{c}.png")
        return

    # Precompute centroids and one-hots for all classes
    centroids = [ _encode_class_centroid(model, loader, c, device) for c in range(num_classes) ]
    y_onehots = [ _to_one_hot(torch.tensor([c], dtype=torch.long, device=device), num_classes, device) for c in range(num_classes) ]

    if args.mode == "interp":
        a = args.class_a
        b = args.class_b if args.class_b >= 0 else (num_classes-1)
        frames = _interp_pair_frames(model, y_onehots[a], y_onehots[b], centroids[a], centroids[b], args.steps)
        _save_gif(frames, f"{args.outdir}/interp_{a}_to_{b}.gif", fps=args.fps)
        return

    if args.mode == "sequence":
        order = list(range(num_classes))
        big_frames = []
        for i in range(len(order)-1):
            a, b = order[i], order[i+1]
            frames = _interp_pair_frames(model, y_onehots[a], y_onehots[b], centroids[a], centroids[b], args.steps)
            big_frames += frames
        _save_gif(big_frames, f"{args.outdir}/sequence_order.gif", fps=args.fps)
        return

    if args.mode == "randomseq":
        order = list(range(num_classes))
        rng = np.random.default_rng(1234)
        rng.shuffle(order)
        big_frames = []
        for i in range(len(order)-1):
            a, b = order[i], order[i+1]
            frames = _interp_pair_frames(model, y_onehots[a], y_onehots[b], centroids[a], centroids[b], args.steps)
            big_frames += frames
        _save_gif(big_frames, f"{args.outdir}/sequence_random.gif", fps=args.fps)
        return


if __name__ == "__main__":
    main()
