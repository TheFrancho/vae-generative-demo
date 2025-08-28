import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.constants import DEFAULT_IMG_SIZE, DEFAULT_NUM_CLASSES
from src.utils import set_seed


def regular_polygon_vertices(k: int, radius: float, phase: float):
    return [(radius * math.cos(2*math.pi*i/k + phase),
             radius * math.sin(2*math.pi*i/k + phase)) for i in range(k)]


def xy_to_px(xy, img_size, cx, cy, scale=1.0, tx=0.0, ty=0.0):
    x, y = xy
    x = (x * scale + tx) * (img_size/2) + cx
    y = (y * scale + ty) * (img_size/2) + cy
    return (x, y)


def draw_shape(img_size: int, k: int, slight: dict, fill: bool = True):
    im = Image.new("L", (img_size, img_size), color=0)
    dr = ImageDraw.Draw(im)
    if k <= 0:  # circle
        r = 0.85 * (1.0 - slight["scale_jitter"])
        cx = cy = img_size/2
        rad = r * (img_size/2)
        bbox = [cx - rad + slight["tx"]*(img_size/2),
                cy - rad + slight["ty"]*(img_size/2),
                cx + rad + slight["tx"]*(img_size/2),
                cy + rad + slight["ty"]*(img_size/2)]
        if fill:
            dr.ellipse(bbox, fill=255)
        else:
            dr.ellipse(bbox, outline=255, width=slight["thickness"])
        return im
    verts = regular_polygon_vertices(k, 0.85 * (1.0 - slight["scale_jitter"]), slight["rotation"])
    px = [xy_to_px(v, img_size, img_size/2, img_size/2, 1.0, slight["tx"], slight["ty"]) for v in verts]
    if fill:
        dr.polygon(px, fill=255)
    else:
        dr.polygon(px, outline=255, width=slight["thickness"])
    return im


def _slight_jitter():
    return {
        "rotation": np.random.uniform(-math.pi/24, math.pi/24),  # ~±7.5°
        "scale_jitter": np.random.uniform(0.00, 0.03),
        "tx": np.random.uniform(-0.02, 0.02),
        "ty": np.random.uniform(-0.02, 0.02),
        "thickness": int(np.random.choice([1, 2]))
    }


def _save_class_grid(Xc: np.ndarray, img_size: int, out_png: str, n_show: int = 25, cols: int = 5):
    n = min(n_show, Xc.shape[0])
    idx = np.random.choice(Xc.shape[0], n, replace=False) if Xc.shape[0] > n else np.arange(n)
    imgs = Xc[idx]  # (n,H,W)
    rows = int(np.ceil(n/cols))
    canvas = np.zeros((rows*img_size, cols*img_size), dtype=np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            if k >= n: break
            y0, y1 = r*img_size, (r+1)*img_size
            x0, x1 = c*img_size, (c+1)*img_size
            canvas[y0:y1, x0:x1] = imgs[k]
            k += 1
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(cols*1.4, rows*1.4))
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES,
                    help="Total classes including circle as last")
    ap.add_argument("--images-per-class", type=int, default=3000)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--out", type=str, default="outputs/datasets/shapes.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fill", action="store_true", help="Filled shapes (default outline)")
    ap.add_argument("--viz-per-class", type=int, default=25, help="samples shown per class grid")
    ap.add_argument("--viz-cols", type=int, default=5, help="columns per class grid")
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    class_to_k = []
    max_poly_sides = 2 + (args.num_classes - 1)  # e.g. 2 + (10 - 1) = 11 -> k in [3..11]
    for k in range(3, max_poly_sides + 1):
        class_to_k.append(k)
    class_to_k.append(0)  # circle
    assert len(class_to_k) == args.num_classes

    N = args.images_per_class * args.num_classes
    X = np.zeros((N, args.img_size, args.img_size), dtype=np.float32)
    y = np.zeros((N,), dtype=np.int64)
    s_cont = np.zeros((N,), dtype=np.float32)

    idx = 0
    per_class_cache = [[] for _ in range(args.num_classes)]
    for ci, k in enumerate(class_to_k):
        for _ in range(args.images_per_class):
            im = draw_shape(args.img_size, k, _slight_jitter(), fill=args.fill)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            X[idx] = arr
            y[idx] = ci
            s_cont[idx] = 0.0 if k == 0 else 1.0 / float(k)
            per_class_cache[ci].append(arr)
            idx += 1

    perm = np.random.permutation(N)
    X, y, s_cont = X[perm], y[perm], s_cont[perm]

    np.savez_compressed(
        args.out,
        X=X, y=y, s=s_cont,
        class_to_k=np.array(class_to_k, dtype=np.int32),
        meta=np.array([f"img_size={args.img_size}",
                       f"num_classes={args.num_classes}",
                       "labels: 0=triangle, ..., last=circle"], dtype=object)
    )
    print(f"Saved dataset: {args.out} | X={X.shape}, y={y.shape}")

    fig_dir = Path("outputs/figures/dataset")
    for ci, imgs in enumerate(per_class_cache):
        Xc = np.stack(imgs, axis=0)
        name = "circle" if class_to_k[ci] == 0 else f"k{class_to_k[ci]}"
        out_png = fig_dir / f"class_{ci}_{name}.png"
        _save_class_grid(Xc, args.img_size, str(out_png), n_show=args.viz_per_class, cols=args.viz_cols)
    print(f"Saved per-class grids to: {fig_dir}")


if __name__ == "__main__":
    main()
