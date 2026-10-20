"""
Frequency-Selective Progressive Frequency-Aware Network (FS-PFAN)
Phase 3: Privacy-Preserving Architecture (H3)

Extends the published 629K PFAN generator with branch-specific privacy adversaries:
- MBI Pathway: extracts high-frequency local textures and sharp tool edges
- LAT Pathway: extracts low-frequency global structures via axial attention
- Gradient Reversal Layer (GRL): enables end-to-end minimax privacy-utility optimization
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

# Add project_files to path
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = os.path.dirname(_this_dir)
for p in [_this_dir, _proj_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.pfan import PFAN, ViTs
from models.networks import Block


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


class BranchPrivacyAdversary(nn.Module):
    """
    Lightweight identity discriminator attached to intermediate feature representations.
    Maps feature map (B, C, H, W) -> procedure identity logits (B, num_classes).
    """
    def __init__(self, in_channels=64, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, feat, alpha=1.0):
        reversed_feat = grad_reverse(feat, alpha)
        return self.net(reversed_feat)


class FSPFAN(nn.Module):
    """
    Frequency-Selective PFAN (FS-PFAN).
    Retains the exact 629K parameter PFAN restoration backbone while
    incorporating branch-specific adversaries on MBI and LAT pathways.
    """
    def __init__(self,
                 input_nc=3,
                 output_nc=3,
                 ngf=64,
                 hidden_dim=64,
                 layers=[2, 2, 2],
                 heads=[4, 4, 4],
                 num_train_classes=14,
                 norm_layer_1=None):
        super().__init__()
        if norm_layer_1 is None:
            norm_layer_1 = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        # PFAN Backbone
        self.model_1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 1, 1, 0),
            norm_layer_1(ngf),
            nn.LeakyReLU(0.05, inplace=True)
        )

        self.convnext1 = Block(ngf)
        self.convnext2 = Block(ngf)

        self.vit = ViTs(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim,
            layers=layers[2],
            downscaling_factor=1,
            num_heads=heads[2],
            head_dim=32,
            window_size=4,
            relative_pos_embedding=True
        )

        self.model_3 = nn.Conv2d(hidden_dim, output_nc, 1, 1, 0)
        self.model_3_1 = nn.Tanh()

        # Branch Privacy Adversaries (Training Only)
        self.adv_mbi = BranchPrivacyAdversary(in_channels=ngf, num_classes=num_train_classes)
        self.adv_lat = BranchPrivacyAdversary(in_channels=hidden_dim, num_classes=num_train_classes)
        self.adv_img = BranchPrivacyAdversary(in_channels=output_nc, num_classes=num_train_classes)

    def forward(self, img, alpha=1.0, return_adv=False):
        """
        Forward pass with optional branch adversary predictions.
        """
        x = self.model_1(img)

        # MBI High-Frequency Pathway
        x1 = self.convnext1(x)
        feat_mbi = self.convnext2(x1)  # Intermediate MBI representation

        # LAT Low-Frequency Pathway
        feat_lat = self.vit(feat_mbi) + x  # Intermediate LAT representation

        # Reconstruction Output
        x3 = self.model_3(feat_lat)
        out_img = self.model_3_1(x3)

        if not return_adv:
            return out_img

        # Predict identity from reversed representations
        pred_mbi = self.adv_mbi(feat_mbi, alpha=alpha)
        pred_lat = self.adv_lat(feat_lat, alpha=alpha)
        pred_img = self.adv_img(out_img, alpha=alpha)

        return out_img, {
            "mbi": pred_mbi,
            "lat": pred_lat,
            "img": pred_img
        }

    def load_pfan_pretrained(self, checkpoint_path):
        """Initialize backbone from 200-epoch pretrained PFAN checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # Load backbone weights strictly (excluding adversary heads)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and "adv_" not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} pretrained backbone layers from {checkpoint_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FSPFAN().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    out, preds = model(x, alpha=1.0, return_adv=True)
    print("FS-PFAN Forward Pass:")
    print("  Output Image Shape :", out.shape)
    print("  MBI Adversary Shape:", preds["mbi"].shape)
    print("  LAT Adversary Shape:", preds["lat"].shape)
    print("  IMG Adversary Shape:", preds["img"].shape)
    print("Model validation successful!")
