"""
Evaluate Privacy Linkage Attacker (H1 Baseline Results)
Phase 1: Privacy Attack Infrastructure & Audit (Reviewer M7 Deliverable).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader


from privacy_attacker import SiameseLinkageAttacker
from linkage_dataset import SurgicalLinkageDataset


def fast_auc(y_true, y_scores):
    """Exact Mann-Whitney U AUC computation via NumPy broadcasting."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def fast_acc(y_true, y_scores):
    """Classification accuracy at 0.5 threshold."""
    preds = (np.asarray(y_scores) >= 0.5).astype(int)
    return float(np.mean(preds == np.asarray(y_true)))


def bootstrap_ci(y_true, y_pred_prob, n_bootstraps=1000, ci=95, seed=42):
    """Compute empirical bootstrap confidence intervals for AUC and Accuracy."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    auc_scores = []
    acc_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        sub_true = y_true[indices]
        sub_prob = y_pred_prob[indices]
        if len(np.unique(sub_true)) < 2:
            continue
        auc_scores.append(fast_auc(sub_true, sub_prob))
        acc_scores.append(fast_acc(sub_true, sub_prob))

    alpha = (100 - ci) / 2.0
    auc_low = np.percentile(auc_scores, alpha)
    auc_high = np.percentile(auc_scores, 100 - alpha)
    acc_low = np.percentile(acc_scores, alpha)
    acc_high = np.percentile(acc_scores, 100 - alpha)

    return (auc_low, auc_high), (acc_low, acc_high)


def evaluate_condition(model, condition, num_pairs=1000, batch_size=32, device="cuda", seed=42):
    """Evaluate privacy linkage attack for a specific image condition on test split."""
    ds = SurgicalLinkageDataset(
        split="test",
        condition=condition,
        num_pairs=num_pairs,
        seed=seed
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x1 = batch["img1"].to(device)
            x2 = batch["img2"].to(device)
            labels = batch["label"].to(device)

            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    acc = fast_acc(all_targets, all_preds)
    auc = fast_auc(all_targets, all_preds)
    (auc_low, auc_high), (acc_low, acc_high) = bootstrap_ci(all_targets, all_preds, seed=seed)

    return {
        "acc": acc,
        "acc_ci": (acc_low, acc_high),
        "auc": auc,
        "auc_ci": (auc_low, auc_high),
        "preds": all_preds,
        "targets": all_targets
    }


def run_h1_evaluation(checkpoint_path=None,
                       desmoked_dir=None,
                       output_dir=None,
                       num_pairs=1000,
                       seed=42):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(_this_dir, "..", "checkpoints", "privacy_attacker", "best_linkage_attacker.pth")
    if desmoked_dir is None:
        desmoked_dir = os.path.join(_this_dir, "..", "results", "pfan_desmoked_test")
    if output_dir is None:
        output_dir = os.path.join(_this_dir, "..", "results")

    checkpoint_path = os.path.abspath(checkpoint_path)
    desmoked_dir = os.path.abspath(desmoked_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running H1 Privacy Linkage Evaluation on {device}...", flush=True)

    model = SiameseLinkageAttacker(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded linkage attacker from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})", flush=True)

    conditions = [
        ("Smoky (Degraded Input)", "smoky"),
        ("Clean (Ground Truth)", "clean"),
        ("PFAN Desmoked (Restored Output)", desmoked_dir)
    ]

    records = []

    for label, cond in conditions:
        print(f"\nAuditing condition: {label}...", flush=True)
        res = evaluate_condition(model, cond, num_pairs=num_pairs, device=device, seed=seed)
        acc_str = f"{res['acc']*100:.2f}% [{res['acc_ci'][0]*100:.2f}%, {res['acc_ci'][1]*100:.2f}%]"
        auc_str = f"{res['auc']:.4f} [{res['auc_ci'][0]:.4f}, {res['auc_ci'][1]:.4f}]"

        print(f"  Linkage Accuracy : {acc_str}", flush=True)
        print(f"  Linkage AUC-ROC  : {auc_str}", flush=True)

        records.append({
            "Frame Condition": label,
            "Linkage Accuracy (%)": round(res["acc"] * 100, 2),
            "Accuracy 95% CI": f"[{res['acc_ci'][0]*100:.2f}%, {res['acc_ci'][1]*100:.2f}%]",
            "Linkage AUC-ROC": round(res["auc"], 4),
            "AUC 95% CI": f"[{res['auc_ci'][0]:.4f}, {res['auc_ci'][1]:.4f}]"
        })

    # Add random chance baseline
    records.append({
        "Frame Condition": "Random Chance Baseline",
        "Linkage Accuracy (%)": 50.00,
        "Accuracy 95% CI": "—",
        "Linkage AUC-ROC": 0.5000,
        "AUC 95% CI": "—"
    })

    csv_path = os.path.join(output_dir, "h1_baseline_results.csv")
    md_path = os.path.join(output_dir, "h1_baseline_results.md")

    fieldnames = ["Frame Condition", "Linkage Accuracy (%)", "Accuracy 95% CI", "Linkage AUC-ROC", "AUC 95% CI"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("# Table I: H1 Baseline Privacy Linkage Audit\n\n")
        fp.write("| Frame Condition | Linkage Accuracy (%) | Accuracy 95% CI | Linkage AUC-ROC | AUC 95% CI |\n")
        fp.write("|:---|:---:|:---:|:---:|:---:|\n")
        for r in records:
            fp.write(f"| **{r['Frame Condition']}** | {r['Linkage Accuracy (%)']} | {r['Accuracy 95% CI']} | {r['Linkage AUC-ROC']} | {r['AUC 95% CI']} |\n")
        fp.write("\n")

    print("\n" + "="*80, flush=True)
    print("=== TABLE I: H1 BASELINE PRIVACY LINKAGE AUDIT (Reviewer M7 Deliverable) ===", flush=True)
    print("="*80, flush=True)
    print(f"{'Frame Condition':<35} | {'Accuracy (%)':<15} | {'AUC-ROC':<15}", flush=True)
    print("-" * 80, flush=True)
    for r in records:
        print(f"{r['Frame Condition']:<35} | {r['Linkage Accuracy (%)']:<6} {r['Accuracy 95% CI']:<10} | {r['Linkage AUC-ROC']:<6} {r['AUC 95% CI']:<10}", flush=True)
    print("="*80, flush=True)
    print(f"Saved results to {csv_path} and {md_path}", flush=True)
    return records



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate H1 privacy leakage.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--desmoked_dir", default=None)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_h1_evaluation(
        checkpoint_path=args.checkpoint,
        desmoked_dir=args.desmoked_dir,
        num_pairs=args.num_pairs,
        seed=args.seed
    )
