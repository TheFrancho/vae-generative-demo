
# Quick commands

Summarized commands from README file to run the full pipeline

## For generating the dataset

```
python -m src.data.generate_shapes \
  --num-classes 5 \
  --images-per-class 3000 \
  --img-size 64 \
  --fill \
  --out outputs/datasets/5_classes.npz
```

## For training a CNN

```
python -m src.train_cnn \
  --data outputs/datasets/5_classes.npz \
  --epochs 10  \
  --batch 128 \
  --outdir outputs/checkpoints/cnn_5_classes
```

## For training a VAE

```
python -m src.train_cvae \
  --data outputs/datasets/5_classes.npz \
  --epochs 15 \
  --batch 256 \
  --z-dim 2 \
  --beta 1.0 \
  --kl-anneal 0.3 \
  --outdir outputs/checkpoints/cvae_5_classes
```

## For plotting the dataset projection

```
python -m src.viz.embed_plot \
  --features outputs/checkpoints/cnn_5_classes/features_val.pt \
  --methods pca tsne umap \
  --data outputs/datasets/5_classes.npz \
  --outdir outputs/figures/embeddings
```

## For making a latent space walking by 2 methods:

### 1. Sequential classes (0 to 9) into a GIF:
```
python -m src.viz.latent_walk \
  --ckpt outputs/checkpoints/cvae_5_classes/cvae_best.pt \
  --data outputs/datasets/5_classes.npz \
  --mode sequence \
  --steps 24 \
  --fps 10 \
  --outdir outputs/figures/latent
```

### 2. Random order across all classes into a GIF:
```
python -m src.viz.latent_walk \
  --ckpt outputs/checkpoints/cvae_5_classes/cvae_best.pt \
  --data outputs/datasets/5_classes.npz \
  --mode randomseq \
  --steps 24 \
  --fps 10 \
  --outdir outputs/figures/latent
```
