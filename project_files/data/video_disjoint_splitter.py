"""
Video Disjoint Splitter
Part of Phase 0: Data Preparation & Video ID Mapping for Laparoscopic Desmoking Privacy.

Addresses Reviewer Feedback M1:
- Splits K=20 clusters into strictly disjoint sets:
  - 14 Train clusters (~460 frames, ~70%)
  - 3 Val clusters (~100 frames, ~15%)
  - 3 Test clusters (~100 frames, ~15%)
- Asserts zero cluster intersection:
  train_clusters ∩ val_clusters ∩ test_clusters = ∅
- Exports metadata/video_splits.json and metadata/frame_metadata.csv.
- Populates datasets/disjoint_splits/{train,val,test} with composite BMPs for drop-in
  compatibility with Pix2Pix / PFAN aligned_dataset.py.
"""

import os
import csv
import json
import shutil
import argparse
import numpy as np


def create_disjoint_splits(clusters_json="project_files/metadata/frame_clusters.json",
                           datasets_dir="project_files/datasets/composite",
                           output_meta_dir="project_files/metadata",
                           output_split_dir="project_files/datasets/disjoint_splits",
                           train_ratio=0.70,
                           val_ratio=0.15,
                           test_ratio=0.15,
                           seed=42):
    if not os.path.exists(clusters_json):
        raise FileNotFoundError(f"Clusters file {clusters_json} not found. Run cholec80_video_mapper.py first.")

    with open(clusters_json, "r") as fp:
        cluster_data = json.load(fp)

    cluster_to_frames = cluster_data["cluster_to_frames"]
    all_clusters = sorted([int(k) for k in cluster_to_frames.keys()])
    n_clusters = len(all_clusters)

    rng = np.random.RandomState(seed)
    shuffled_clusters = rng.permutation(all_clusters).tolist()

    n_train = int(round(n_clusters * train_ratio))
    n_val = int(round(n_clusters * val_ratio))
    # Remaining go to test
    n_test = n_clusters - n_train - n_val

    train_clusters = sorted(shuffled_clusters[:n_train])
    val_clusters = sorted(shuffled_clusters[n_train:n_train + n_val])
    test_clusters = sorted(shuffled_clusters[n_train + n_val:])

    # Strictly verify disjointness
    set_train = set(train_clusters)
    set_val = set(val_clusters)
    set_test = set(test_clusters)

    assert len(set_train & set_val) == 0, "Train and Val clusters overlap!"
    assert len(set_train & set_test) == 0, "Train and Test clusters overlap!"
    assert len(set_val & set_test) == 0, "Val and Test clusters overlap!"
    assert len(set_train | set_val | set_test) == n_clusters, "Missing clusters in split!"

    print("=== Disjoint Cluster Split Verification ===")
    print(f"Total Clusters: {n_clusters}")
    print(f"Train Clusters ({len(train_clusters)}): {train_clusters}")
    print(f"Val Clusters   ({len(val_clusters)}): {val_clusters}")
    print(f"Test Clusters  ({len(test_clusters)}): {test_clusters}")

    # Gather frame assignments
    splits = {"train": [], "val": [], "test": []}
    cluster_split_map = {}

    for c in train_clusters:
        cluster_split_map[c] = "train"
        splits["train"].extend([(f, c) for f in cluster_to_frames[str(c)]])

    for c in val_clusters:
        cluster_split_map[c] = "val"
        splits["val"].extend([(f, c) for f in cluster_to_frames[str(c)]])

    for c in test_clusters:
        cluster_split_map[c] = "test"
        splits["test"].extend([(f, c) for f in cluster_to_frames[str(c)]])

    print(f"Frame Counts: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}, Total={len(splits['train']) + len(splits['val']) + len(splits['test'])}")

    # Write video_splits.json
    splits_meta_file = os.path.join(output_meta_dir, "video_splits.json")
    with open(splits_meta_file, "w") as fp:
        json.dump({
            "seed": seed,
            "train_clusters": train_clusters,
            "val_clusters": val_clusters,
            "test_clusters": test_clusters,
            "train_frames_count": len(splits["train"]),
            "val_frames_count": len(splits["val"]),
            "test_frames_count": len(splits["test"]),
            "frame_counts_per_cluster": {c: len(cluster_to_frames[str(c)]) for c in all_clusters}
        }, fp, indent=2)

    # Write frame_metadata.csv
    csv_file = os.path.join(output_meta_dir, "frame_metadata.csv")
    with open(csv_file, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame_id", "bmp_name", "cluster_id", "split", "trainA_path", "trainB_path", "composite_path"])

        all_entries = []
        for split_name in ["train", "val", "test"]:
            for f, c in splits[split_name]:
                frame_id = int(os.path.splitext(f)[0])
                all_entries.append((
                    frame_id,
                    f,
                    c,
                    split_name,
                    os.path.join(datasets_dir, "trainA", f),
                    os.path.join(datasets_dir, "trainB", f),
                    os.path.join(datasets_dir, "train", f)
                ))

        all_entries.sort(key=lambda x: x[0])
        for entry in all_entries:
            writer.writerow(entry)

    print(f"Saved split metadata to {splits_meta_file} and {csv_file}")

    # Build disjoint physical directories for drop-in dataloader compatibility
    os.makedirs(os.path.join(output_split_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_split_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_split_dir, "test"), exist_ok=True)

    print(f"Populating physical disjoint split folders at {output_split_dir}...")
    composite_source = os.path.join(datasets_dir, "train")

    for split_name in ["train", "val", "test"]:
        dst_folder = os.path.join(output_split_dir, split_name)
        for f, _ in splits[split_name]:
            src_file = os.path.join(composite_source, f)
            dst_file = os.path.join(dst_folder, f)
            if not os.path.exists(dst_file) and os.path.exists(src_file):
                shutil.copyfile(src_file, dst_file)

    print("Physical disjoint folders successfully prepared:")
    print(f"  {output_split_dir}/train: {len(os.listdir(os.path.join(output_split_dir, 'train')))} files")
    print(f"  {output_split_dir}/val:   {len(os.listdir(os.path.join(output_split_dir, 'val')))} files")
    print(f"  {output_split_dir}/test:  {len(os.listdir(os.path.join(output_split_dir, 'test')))} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video-disjoint splits.")
    parser.add_argument("--clusters_json", default="project_files/metadata/frame_clusters.json")
    parser.add_argument("--datasets_dir", default="project_files/datasets/composite")
    parser.add_argument("--output_meta_dir", default="project_files/metadata")
    parser.add_argument("--output_split_dir", default="project_files/datasets/disjoint_splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_disjoint_splits(
        clusters_json=args.clusters_json,
        datasets_dir=args.datasets_dir,
        output_meta_dir=args.output_meta_dir,
        output_split_dir=args.output_split_dir,
        seed=args.seed
    )
