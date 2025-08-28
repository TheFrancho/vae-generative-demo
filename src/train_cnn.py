import argparse
from pathlib import Path

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from src.data.dataset import make_loaders
from src.models.cnn_classifier import SmallCNN
from src.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="outputs/datasets/shapes.npz")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="outputs/ckpts/cnn")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    train_Dataloader, val_dataloader, _ = make_loaders(args.data, batch_size=args.batch)

    num_classes = train_Dataloader.dataset.num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(num_classes).to(device)
    opt = Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_msg = ""
    last_val = None

    for epoch in range(1, args.epochs+1):
        model.train()
        train_losses = []
        for x,y,_ in tqdm(train_Dataloader, desc=f"train epoch {epoch}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses)/len(train_losses)

        model.eval()
        val_losses, y_true, y_pred = [], [], []
        with torch.no_grad():
            for x,y,_ in val_dataloader:
                x,y = x.to(device), y.to(device)
                logits, feats = model(x)
                loss = crit(logits, y)
                val_losses.append(loss.item())
                y_true.append(y.cpu())
                y_pred.append(logits.argmax(1).cpu())

        val_loss = sum(val_losses)/len(val_losses)
        last_val = val_loss
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        f1_macro = f1_score(y_true, y_pred, average="macro")

        # Save features when best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.outdir}/cnn_best.pt")
            # save features/labels for embed plot
            feats_list = []
            with torch.no_grad():
                for x,y,_ in val_dataloader:
                    x = x.to(device)
                    _, feats = model(x)
                    feats_list.append(feats.cpu())
            torch.save({"features": torch.cat(feats_list), "labels": torch.tensor(y_true)},
                       f"{args.outdir}/features_val.pt")
            best_msg = f"[BEST] epoch {epoch}: val_loss={val_loss:.4f}, f1_macro={f1_macro:.4f} -> saved cnn_best.pt & features_val.pt"
            print(best_msg)
        else:
            print(f"epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | f1_macro={f1_macro:.4f}")

    print(f"\nFinal: last_val_loss={last_val:.4f} | best_val_loss={best_val:.4f}")
    if best_msg: print(best_msg)

if __name__ == "__main__":
    main()
