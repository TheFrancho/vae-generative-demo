import torch.nn as nn


class SmallCNN(nn.Module):
    '''
    Simple CNN for demonstrating data convergence
    '''
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            nn.Conv2d(32,64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(64,128,3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        logits = self.head(feat)
        return logits, feat
