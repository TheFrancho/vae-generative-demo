from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ShapesDataset(Dataset):
    def __init__(self, npz_path: str, split: str = "train", augment: bool = False,
                 one_hot: bool = False, num_classes: Optional[int] = None, seed: int = 42):
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"] # (N,H,W)
        y = data["y"] # (N,)
        s = data["s"] # (N,)
        N = X.shape[0]

        # 80/10/10 split
        rng = np.random.default_rng(seed)
        idx = np.arange(N)
        rng.shuffle(idx)
        n_train = int(0.8 * N)
        n_val = int(0.1 * N)
        splits = {
            "train": idx[:n_train],
            "val": idx[n_train:n_train+n_val],
            "test": idx[n_train+n_val:]
        }
        sel = splits[split]
        self.X = X[sel]
        self.y = y[sel]
        self.s = s[sel]
        self.augment = augment
        self.one_hot = one_hot
        self.num_classes = num_classes or int(y.max() + 1)

    def __len__(self):
        return self.X.shape[0]

    def _gentle_aug(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1,H,W) in [0,1]
        B, C, H, W = 1, 1, x.shape[-2], x.shape[-1]
        angle = torch.empty(1).uniform_(-3.0, 3.0) * (3.14159/180.0)
        tx = torch.empty(1).uniform_(-0.02, 0.02)
        ty = torch.empty(1).uniform_(-0.02, 0.02)
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.tensor([[[ cos.item(), -sin.item(), tx.item()],
                               [ sin.item(),  cos.item(), ty.item()]]], dtype=torch.float32)
        grid = torch.nn.functional.affine_grid(theta, size=(1,1,H,W), align_corners=False)
        x = torch.nn.functional.grid_sample(x.unsqueeze(0), grid, align_corners=False)
        return x.squeeze(0)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.X[i])[None, ...]
        y = int(self.y[i])
        s = float(self.s[i])
        if self.augment:
            x = self._gentle_aug(x)
        if self.one_hot:
            y_oh = torch.zeros(self.num_classes, dtype=torch.float32)
            y_oh[y] = 1.0
            return x, y, y_oh, torch.tensor(s, dtype=torch.float32)
        return x, y, torch.tensor(s, dtype=torch.float32)


def make_loaders(npz_path: str, batch_size: int = 128, num_workers: int = 2,
                 one_hot_for_cvae: bool = False):
    ds_tr = ShapesDataset(npz_path, split="train", augment=True, one_hot=one_hot_for_cvae)
    ds_va = ShapesDataset(npz_path, split="val",   augment=False, one_hot=one_hot_for_cvae)
    ds_te = ShapesDataset(npz_path, split="test",  augment=False, one_hot=one_hot_for_cvae)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_tr, dl_va, dl_te
