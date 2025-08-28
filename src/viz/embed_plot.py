import argparse
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import umap


def _save_scatter(X2, y, outpath, title, legend_labels=None, add_legend=True, cmap_name="tab10"):
    plt.figure(figsize=(6.6, 5.4))

    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=6, alpha=0.85, cmap=cmap_name)
    plt.title(title)
    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    if add_legend and legend_labels is not None:
        classes = np.unique(y)
        handles = []
        cmap = scatter.cmap
        norm = scatter.norm
        for c in classes:
            label = legend_labels.get(int(c), str(c))
            color = cmap(norm(c))
            handles.append(
                Line2D([0], [0],
                       marker='o', linestyle='',
                       markersize=6,
                       markerfacecolor=color,
                       markeredgecolor=color,
                       label=label)
            )
        leg = plt.legend(handles=handles, title="Classes", loc="best", frameon=True)
        leg._legend_box.align = "left"
    plt.savefig(outpath, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True, help="Path to features .pt saved by train_cnn")
    ap.add_argument("--methods", nargs="+", default=["pca", "tsne", "umap"], choices=["pca","tsne","umap"])
    ap.add_argument("--outdir", type=str, default="outputs/figures/embeddings")
    ap.add_argument("--pca-components", type=int, default=50, help="PCA pre-reduction for t-SNE/UMAP")
    ap.add_argument("--data", type=str, default=None, help="Optional: dataset npz to show class labels in legend")
    ap.add_argument("--no-legend", action="store_true", help="Disable legend")
    ap.add_argument("--cmap", type=str, default="tab10", help="Matplotlib colormap (e.g., tab10, tab20, viridis)")
    args = ap.parse_args()

    blob = torch.load(args.features)
    X = blob["features"].numpy()
    y = blob["labels"].numpy()

    legend_labels = None
    if args.data:
        dat = np.load(args.data, allow_pickle=True)
        class_to_k = dat["class_to_k"]
        legend_labels = {}
        for idx, k in enumerate(class_to_k):
            legend_labels[idx] = "circle" if int(k) == 0 else f"k={int(k)}"

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if "pca" in args.methods:
        p = PCA(n_components=2).fit_transform(X)
        _save_scatter(p, y, f"{args.outdir}/pca.png", "PCA (2D)", legend_labels, not args.no_legend, cmap_name=args.cmap)

    Xp = PCA(n_components=min(args.pca_components, X.shape[1])).fit_transform(X)

    if "tsne" in args.methods:
        t = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca").fit_transform(Xp)
        _save_scatter(t, y, f"{args.outdir}/tsne.png", "t-SNE (2D)", legend_labels, not args.no_legend, cmap_name=args.cmap)

    if "umap" in args.methods:
        u = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1).fit_transform(Xp)
        _save_scatter(u, y, f"{args.outdir}/umap.png", "UMAP (2D)", legend_labels, not args.no_legend, cmap_name=args.cmap)

    print(f"Saved plots to: {args.outdir}")

if __name__ == "__main__":
    main()