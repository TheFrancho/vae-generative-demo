import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.data.dataset import make_loaders
from src.models.cvae import CVAE
from src.utils import set_seed


def loss_vae(x_hat, x, mu, logvar, beta=1.0):
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta*kl, recon, kl


def dice_score(x_hat_bin, x_bin, eps=1e-8):
    inter = (x_hat_bin * x_bin).sum(dim=(1,2,3))
    denom = x_hat_bin.sum(dim=(1,2,3)) + x_bin.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (denom + eps)
    return dice.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="outputs/datasets/shapes.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--z-dim", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--kl-anneal", type=float, default=0.3, help="fraction of steps to ramp beta")
    ap.add_argument("--outdir", type=str, default="outputs/ckpts/cvae")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, _ = make_loaders(args.data, batch_size=args.batch, one_hot_for_cvae=True)
    num_classes = train_dataloader.dataset.num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(num_classes=num_classes, z_dim=args.z_dim).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(train_dataloader)
    step = 0
    best_val = float("inf")
    best_msg = ""
    last_val = None

    for ep in range(1, args.epochs+1):
        model.train()
        tr_losses, tr_recons, tr_kls = [], [], []
        for x, _, y_oh, _ in tqdm(train_dataloader, desc=f"train epoch {ep}"):
            x, y_oh = x.to(device), y_oh.to(device)
            step += 1
            frac = min(1.0, step / max(1, int(args.kl_anneal * total_steps)))
            beta = args.beta * frac
            opt.zero_grad()
            x_hat, mu, logvar = model(x, y_oh)
            loss, recon, kl = loss_vae(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
            tr_recons.append(recon.item())
            tr_kls.append(kl.item())
        tr_elbo = sum(tr_losses) / len(tr_losses)
        tr_recon = sum(tr_recons) / len(tr_recons)
        tr_kl = sum(tr_kls) / len(tr_kls)

        model.eval()
        va_losses, va_recons, va_kls = [], [], []
        dices = []
        with torch.no_grad():
            for x, _, y_oh, _ in val_dataloader:
                x, y_oh = x.to(device), y_oh.to(device)
                x_hat, mu, logvar = model(x, y_oh)
                l, r, k = loss_vae(x_hat, x, mu, logvar, beta=args.beta)
                va_losses.append(l.item())
                va_recons.append(r.item())
                va_kls.append(k.item())

                xb = (x > 0.5).float()
                xhb = (x_hat > 0.5).float()
                dices.append(dice_score(xhb, xb).item())

        va_elbo = sum(va_losses) / len(va_losses)
        va_recon = sum(va_recons) / len(va_recons)
        va_kl = sum(va_kls) / len(va_kls)
        va_dice = sum(dices) / len(dices)
        last_val = va_elbo

        if va_elbo < best_val:
            best_val = va_elbo
            torch.save(model.state_dict(), f"{args.outdir}/cvae_best.pt")
            best_msg = (f"[BEST] epoch {ep}: val_ELBO={va_elbo:.4f} (recon={va_recon:.4f}, KL={va_kl:.4f}), "
                        f"val_Dice={va_dice:.4f} -> saved cvae_best.pt")
            print(best_msg)
        else:
            print(f"epoch {ep}: "
                  f"train_ELBO={tr_elbo:.4f} (recon={tr_recon:.4f}, KL={tr_kl:.4f}) | "
                  f"val_ELBO={va_elbo:.4f} (recon={va_recon:.4f}, KL={va_kl:.4f}) | "
                  f"val_Dice={va_dice:.4f}")

    print(f"\nFinal: last_val_ELBO={last_val:.4f} | best_val_ELBO={best_val:.4f}")
    if best_msg: print(best_msg)

if __name__ == "__main__":
    main()
