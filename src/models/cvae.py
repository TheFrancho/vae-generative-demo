import torch
import torch.nn as nn


class CVAE(nn.Module):
    """
    Conditional VAE:
      - Encoder sees image and label (one-hot)
      - Decoder receives latent z and label (one-hot)
    """
    def __init__(self, num_classes: int, z_dim: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim

        # fuse label by channel-wise concat (tile to HxW)
        def enc_block(in_ch, out_ch):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())
        self.enc = nn.Sequential(
            enc_block(1 + 1, 32),
            nn.MaxPool2d(2),
            enc_block(32, 64),
            nn.MaxPool2d(2),
            enc_block(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

        # decoder: project (z + y) -> spatial
        self.dec_in = nn.Linear(z_dim + num_classes, 128*8*8)
        def dec_block(in_ch, out_ch):
            return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                                 nn.BatchNorm2d(out_ch), nn.ReLU())
        self.dec = nn.Sequential(
            dec_block(128, 64), # 16x16
            dec_block(64, 32), # 32x32
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), # 64x64
        )

    def encode(self, x, y_oh):
        B, _, H, W = x.shape
        y_map = y_oh.view(B, self.num_classes, 1, 1) # (B,C,1,1)
        y_map = y_map.mean(dim=1, keepdim=True) # compress to 1 channel -> (B,1,1,1)
        y_map = y_map.expand(B,1,H,W) # broadcast
        h = self.enc(torch.cat([x, y_map], dim=1)).flatten(1)
        mu = self.mu(h); logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_oh):
        zy = torch.cat([z, y_oh], dim=1)
        h = self.dec_in(zy).view(-1, 128, 8, 8)
        x_hat = self.dec(h)
        return torch.sigmoid(x_hat)

    def forward(self, x, y_oh):
        mu, logvar = self.encode(x, y_oh)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y_oh)
        return x_hat, mu, logvar
