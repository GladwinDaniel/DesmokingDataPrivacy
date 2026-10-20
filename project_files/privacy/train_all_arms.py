"""
Master Script: Train All FS-PFAN Arms & Ablations (Phases 3 & 4)
Runs fine-tuning for all arms sequentially with RAM caching.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = os.path.dirname(_this_dir)
_root_dir = os.path.dirname(_proj_dir)
for p in [_this_dir, _proj_dir, _root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import time
from privacy.train_fs_pfan import train_arm


def run_all():
    experiments = [
        # Arm 1: Full Joint FS-PFAN (Proposed)
        {"arm_name": "joint_fs_pfan", "mbi": 0.025, "lat": 0.025, "img": 0.0, "epochs": 3},

        # Arm 2: MBI-Only Frequency Adversary (High-Freq Ablation)
        {"arm_name": "mbi_only", "mbi": 0.05, "lat": 0.0, "img": 0.0, "epochs": 3},

        # Arm 3: LAT-Only Frequency Adversary (Low-Freq Ablation)
        {"arm_name": "lat_only", "mbi": 0.0, "lat": 0.05, "img": 0.0, "epochs": 3},

        # Arm 4: Image-Level Adversary Baseline
        {"arm_name": "img_only", "mbi": 0.0, "lat": 0.0, "img": 0.05, "epochs": 3},

        # Arm 5: Pareto Frontier: Low Budget (lambda=0.01)
        {"arm_name": "pareto_0.01", "mbi": 0.005, "lat": 0.005, "img": 0.0, "epochs": 3},

        # Arm 6: Pareto Frontier: Moderate Budget (lambda=0.10)
        {"arm_name": "pareto_0.10", "mbi": 0.05, "lat": 0.05, "img": 0.0, "epochs": 3},

        # Arm 7: Pareto Frontier: High Budget (lambda=0.20)
        {"arm_name": "pareto_0.20", "mbi": 0.10, "lat": 0.10, "img": 0.0, "epochs": 3},
    ]

    print("========================================================")
    print(f"Starting Training for {len(experiments)} Experimental Arms")
    print("========================================================")

    results = {}
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n>>> Running Experiment {i}/{len(experiments)}: {exp['arm_name']}")
        ckpt = train_arm(
            arm_name=exp["arm_name"],
            lambda_mbi=exp["mbi"],
            lambda_lat=exp["lat"],
            lambda_img=exp["img"],
            epochs=exp["epochs"],
            batch_size=2
        )
        results[exp["arm_name"]] = ckpt

    elapsed = time.time() - total_start
    print("\n========================================================")
    print(f"All {len(experiments)} Arms Successfully Trained in {elapsed/60.0:.2f} minutes!")
    print("Checkpoints:")
    for k, v in results.items():
        print(f"  {k:15s} -> {v}")
    print("========================================================")


if __name__ == "__main__":
    run_all()
