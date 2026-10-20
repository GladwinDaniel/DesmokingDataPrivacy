"""
Linkage Dataset for Siamese Privacy Attacker
Open-Set Surgical Procedure Linkage Evaluation (Reviewer M3).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def get_default_metadata_csv():
    candidates = [
        "project_files/metadata/frame_metadata.csv",
        "metadata/frame_metadata.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metadata", "frame_metadata.csv")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return "project_files/metadata/frame_metadata.csv"


class SurgicalLinkageDataset(Dataset):
    def __init__(self,
                 metadata_csv=None,
                 split="train",
                 condition="clean",  # 'clean', 'smoky', or directory path to restored images
                 num_pairs=1000,
                 img_size=256,
                 seed=42):
        super().__init__()
        if metadata_csv is None:
            metadata_csv = get_default_metadata_csv()

        self.split = split
        self.condition = condition
        self.num_pairs = num_pairs
        self.rng = random.Random(seed)

        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file {metadata_csv} not found.")

        with open(metadata_csv, "r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            split_rows = [r for r in reader if r["split"] == split]

        if len(split_rows) == 0:
            raise ValueError(f"No frames found for split '{split}' in {metadata_csv}")

        # Group frames by cluster
        self.cluster_to_frames = {}
        self.frame_to_path = {}

        for row in split_rows:
            c = int(row["cluster_id"])
            bmp = row["bmp_name"]

            if c not in self.cluster_to_frames:
                self.cluster_to_frames[c] = []
            self.cluster_to_frames[c].append(bmp)

            if condition == "clean":
                self.frame_to_path[bmp] = row["trainB_path"]
            elif condition == "smoky":
                self.frame_to_path[bmp] = row["trainA_path"]
            else:
                # Custom restored directory (or file map)
                if os.path.isdir(condition):
                    self.frame_to_path[bmp] = os.path.join(condition, bmp)
                else:
                    self.frame_to_path[bmp] = row["trainB_path"]


        self.clusters = sorted(list(self.cluster_to_frames.keys()))

        # Pre-generate deterministic pairs for reproducibility
        self.pairs = self._build_pairs()

        # Image transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def _build_pairs(self):
        """Build balanced positive (y=1) and negative (y=0) pairs."""
        pairs = []
        n_pos = self.num_pairs // 2
        n_neg = self.num_pairs - n_pos

        # Positive pairs: same cluster, distinct frames
        eligible_pos_clusters = [c for c in self.clusters if len(self.cluster_to_frames[c]) >= 2]
        if not eligible_pos_clusters:
            raise ValueError(f"Split {self.split} has no clusters with >= 2 frames for positive pairs!")

        for _ in range(n_pos):
            c = self.rng.choice(eligible_pos_clusters)
            f1, f2 = self.rng.sample(self.cluster_to_frames[c], 2)
            pairs.append((f1, f2, 1.0, c, c))

        # Negative pairs: different clusters
        for _ in range(n_neg):
            c1, c2 = self.rng.sample(self.clusters, 2)
            f1 = self.rng.choice(self.cluster_to_frames[c1])
            f2 = self.rng.choice(self.cluster_to_frames[c2])
            pairs.append((f1, f2, 0.0, c1, c2))

        # Shuffle pairs
        self.rng.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2, label, c1, c2 = self.pairs[idx]
        p1 = self.frame_to_path[f1]
        p2 = self.frame_to_path[f2]

        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")

        tensor1 = self.transform(im1)
        tensor2 = self.transform(im2)

        return {
            "img1": tensor1,
            "img2": tensor2,
            "label": torch.tensor(label, dtype=torch.float32),
            "cluster1": torch.tensor(c1, dtype=torch.long),
            "cluster2": torch.tensor(c2, dtype=torch.long),
            "file1": f1,
            "file2": f2
        }


if __name__ == "__main__":
    print("Testing SurgicalLinkageDataset...")
    ds_train = SurgicalLinkageDataset(split="train", condition="clean", num_pairs=20)
    print("Train dataset length:", len(ds_train))
    sample = ds_train[0]
    print("Sample img1 shape:", sample["img1"].shape)
    print("Sample label:", sample["label"].item())
    print("Success!")
