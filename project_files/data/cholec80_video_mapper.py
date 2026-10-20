"""
Cholec80 Surgical Scene / Video Cluster Mapper
Part of Phase 0: Data Preparation & Video ID Mapping for Laparoscopic Desmoking Privacy.

Addresses Reviewer Feedback M1 & M2:
- Groups the 660 Cholec80 frames into K=20 robust surgical procedure / scene clusters
  with ~30-50 frames per cluster (avoiding single-digit sparsity overfitting).
- Uses deep ResNet-18 perceptual embeddings + L2 normalization + KMeans.
- Exports cluster mappings to metadata/frame_clusters.json.
"""

import os
import json
import argparse
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def extract_features(image_dir, device='cuda', batch_size=32):
    """Extract L2-normalized ResNet-18 features for all clean BMP frames."""
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.bmp')])
    if not files:
        raise FileNotFoundError(f"No BMP images found in {image_dir}")

    print(f"Loading ResNet-18 backbone on {device}...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    features = []
    print(f"Extracting perceptual embeddings for {len(files)} frames...")
    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            tensors = []
            for f in batch_files:
                im = Image.open(os.path.join(image_dir, f)).convert('RGB')
                tensors.append(transform(im))
            batch_tensor = torch.stack(tensors).to(device)
            feats = model(batch_tensor).cpu().numpy()
            for feat in feats:
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
                features.append(feat)

    features = np.array(features)
    print(f"Extracted feature matrix: {features.shape}")
    return files, features


def cluster_frames(files, features, n_clusters=20, random_state=42):
    """Cluster frames into n_clusters surgical scene clusters."""
    print(f"Performing KMeans clustering with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    sil = silhouette_score(features, labels)
    counts = np.bincount(labels)

    print(f"Clustering complete. Silhouette Score: {sil:.4f}")
    print(f"Cluster sizes: min={counts.min()}, median={np.median(counts):.1f}, max={counts.max()}, std={np.std(counts):.1f}")

    mapping = {}
    for f, label in zip(files, labels):
        mapping[f] = int(label)

    cluster_to_files = {int(k): [] for k in range(n_clusters)}
    for f, label in zip(files, labels):
        cluster_to_files[int(label)].append(f)

    return mapping, cluster_to_files, sil, counts.tolist()


def run_mapping(trainB_dir="project_files/datasets/composite/trainB",
                output_dir="project_files/metadata",
                n_clusters=20,
                seed=42):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    files, features = extract_features(trainB_dir, device=device)
    frame_to_cluster, cluster_to_frames, sil, counts = cluster_frames(
        files, features, n_clusters=n_clusters, random_state=seed
    )

    out_file = os.path.join(output_dir, "frame_clusters.json")
    with open(out_file, "w") as fp:
        json.dump({
            "n_clusters": n_clusters,
            "silhouette_score": float(sil),
            "cluster_counts": counts,
            "frame_to_cluster": frame_to_cluster,
            "cluster_to_frames": cluster_to_frames
        }, fp, indent=2)

    # Save feature matrix for fast downstream reuse
    feat_file = os.path.join(output_dir, "clean_frame_embeddings.npy")
    np.save(feat_file, features)
    print(f"Saved cluster mapping to {out_file} and embeddings to {feat_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Cholec80 frames into video clusters.")
    parser.add_argument("--trainB_dir", default="project_files/datasets/composite/trainB", help="Path to clean frames")
    parser.add_argument("--output_dir", default="project_files/metadata", help="Metadata output dir")
    parser.add_argument("--n_clusters", type=int, default=20, help="Number of clusters (Reviewer M2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_mapping(args.trainB_dir, args.output_dir, args.n_clusters, args.seed)
