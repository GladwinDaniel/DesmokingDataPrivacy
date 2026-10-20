"""
Siamese Privacy Linkage Attacker Network
Phase 1: Privacy Attack Infrastructure (H1)

Addresses Reviewer Feedback M3 & M7:
- Formulated as an open-set linkage verifier: given two images (x_A, x_B),
  predicts whether they originate from the same surgical procedure / video.
- Backbone: Pretrained ResNet-18 (feature dim = 512).
- Pairwise feature interaction: concatenates absolute difference |e_A - e_B|
  and element-wise product e_A * e_B (1024-d representation).
- Multi-layer classification head with dropout regularization.
"""

import torch
import torch.nn as nn
from torchvision import models


class SiameseLinkageAttacker(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone_layers=0):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = 512

        # Optional layer freezing
        if freeze_backbone_layers > 0:
            for i, child in enumerate(self.backbone.children()):
                if i < freeze_backbone_layers:
                    for param in child.parameters():
                        param.requires_grad = False

        # Interaction Head: (|e_A - e_B|, e_A * e_B) -> 1024
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def extract_features(self, x):
        """Extract L2-normalized 512-d feature vector for a batch of images."""
        feat = self.backbone(x)
        feat = torch.flatten(feat, 1)
        feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat

    def forward_pairs(self, feat_a, feat_b):
        """Compute linkage logit given two pre-extracted feature tensors."""
        diff = torch.abs(feat_a - feat_b)
        prod = feat_a * feat_b
        combined = torch.cat([diff, prod], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(1)

    def forward(self, x_a, x_b):
        """Full forward pass for a pair of input images."""
        feat_a = self.extract_features(x_a)
        feat_b = self.extract_features(x_b)
        return self.forward_pairs(feat_a, feat_b)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseLinkageAttacker().to(device)
    x1 = torch.randn(4, 3, 256, 256).to(device)
    x2 = torch.randn(4, 3, 256, 256).to(device)
    out = model(x1, x2)
    print("Linkage Attacker forward pass output shape:", out.shape)
    probs = torch.sigmoid(out)
    print("Predicted probabilities:", probs.tolist())
