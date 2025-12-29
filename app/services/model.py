import torch
import torch.nn as nn
from torchvision import models
# from .config import NUM_CLASSES
NUM_CLASSES = 6

class SkinNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Load backbone EfficientNet B0
        self.cnn = models.efficientnet_b0(weights=None) 
        
        # Mở khóa các layer 
        for param in self.cnn.parameters():
            param.requires_grad = True

        # Thay thế classifier
        in_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()

        # Nhánh ảnh
        self.image_fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5) 
        )

        # Nhánh Metadata
        self.meta_fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Fusion
        self.classifier = nn.Linear(128 + 32, num_classes)

    def forward(self, image, meta):
        img_feat = self.cnn(image)
        img_feat = self.image_fc(img_feat)
        meta_feat = self.meta_fc(meta)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(fused)