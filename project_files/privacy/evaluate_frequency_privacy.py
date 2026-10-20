"""
Evaluate Frequency-Subband Privacy Leakage (H2 Results)
Phase 2: Frequency-Domain Privacy Analysis.

Evaluates the trained Siamese Linkage Attacker on LL, LH, HL, HH sub-bands
and compares against the full spectrum PFAN desmoked output.
Computes Accuracy, AUC-ROC, and 95% Bootstrap Confidence Intervals.
Outputs the canonical H2 results table.
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
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def fast_acc(y_true, y_scores):
    preds = (np.asarray(y_scores) >= 0.5).astype(int)
    return float(np.mean(preds == np.asarray(y_true)))


def bootstrap_ci(y_true, y_pred_prob, n_bootstraps=1000, ci=95, seed=42):
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


def evaluate_subband(model, folder_path, num_pairs=1000, batch_size=32, device="cuda", seed=42):
    ds = SurgicalLinkageDataset(
        split="test",
        condition=folder_path,
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
        "auc_ci": (auc_low, auc_high)
    }


def run_h2_evaluation(checkpoint_path=None,
                       subbands_base=None,
                       full_desmoked_dir=None,
                       output_dir=None,
                       num_pairs=1000,
                       seed=42):
    if checkpoint_path is None:
        checkpoint_path = os.path.abspath(os.path.join(_this_dir, "..", "checkpoints", "privacy_attacker", "best_linkage_attacker.pth"))
    if subbands_base is None:
        subbands_base = os.path.abspath(os.path.join(_this_dir, "..", "results", "frequency_subbands"))
    if full_desmoked_dir is None:
        full_desmoked_dir = os.path.abspath(os.path.join(_this_dir, "..", "results", "pfan_desmoked_test"))
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(_this_dir, "..", "results"))

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running H2 Frequency-Subband Privacy Evaluation on {device}...", flush=True)

    model = SiameseLinkageAttacker(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    subband_configs = [
        ("LL (Low-Frequency Approx)", os.path.join(subbands_base, "ll"), "Macroscopic tissue color & illumination"),
        ("LH (Horizontal Detail)", os.path.join(subbands_base, "lh"), "Horizontal boundaries & organ contours"),
        ("HL (Vertical Detail)", os.path.join(subbands_base, "hl"), "Vertical tissue structures & smoke edges"),
        ("HH (Diagonal Detail)", os.path.join(subbands_base, "hh"), "Fine tissue texture & microsurgical marks"),
        ("Full Spectrum (Baseline)", full_desmoked_dir, "Complete PFAN restored frame")
    ]

    records = []

    for name, path, content_desc in subband_configs:
        print(f"Auditing sub-band: {name}...", flush=True)
        res = evaluate_subband(model, path, num_pairs=num_pairs, device=device, seed=seed)
        acc_str = f"{res['acc']*100:.2f}% [{res['acc_ci'][0]*100:.2f}%, {res['acc_ci'][1]*100:.2f}%]"
        auc_str = f"{res['auc']:.4f} [{res['auc_ci'][0]:.4f}, {res['auc_ci'][1]:.4f}]"

        print(f"  Accuracy : {acc_str}", flush=True)
        print(f"  AUC-ROC  : {auc_str}", flush=True)

        records.append({
            "Frequency Band": name,
            "Clinical Content": content_desc,
            "Linkage Accuracy (%)": round(res["acc"] * 100, 2),
            "Accuracy 95% CI": f"[{res['acc_ci'][0]*100:.2f}%, {res['acc_ci'][1]*100:.2f}%]",
            "Linkage AUC-ROC": round(res["auc"], 4),
            "AUC 95% CI": f"[{res['auc_ci'][0]:.4f}, {res['auc_ci'][1]:.4f}]"
        })

    # Add Random Chance Baseline
    records.append({
        "Frequency Band": "Random Chance Baseline",
        "Clinical Content": "Theoretical uninformative floor",
        "Linkage Accuracy (%)": 50.00,
        "Accuracy 95% CI": "—",
        "Linkage AUC-ROC": 0.5000,
        "AUC 95% CI": "—"
    })

    csv_path = os.path.join(output_dir, "h2_frequency_results.csv")
    md_path = os.path.join(output_dir, "h2_frequency_results.md")

    fieldnames = ["Frequency Band", "Clinical Content", "Linkage Accuracy (%)", "Accuracy 95% CI", "Linkage AUC-ROC", "AUC 95% CI"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("# Table II: H2 Frequency-Domain Privacy Leakage Breakdown\n\n")
        fp.write("| Frequency Band | Clinical Content | Linkage Accuracy (%) | Accuracy 95% CI | Linkage AUC-ROC | AUC 95% CI |\n")
        fp.write("|:---|:---|:---:|:---:|:---:|:---:|\n")
        for r in records:
            fp.write(f"| **{r['Frequency Band']}** | {r['Clinical Content']} | {r['Linkage Accuracy (%)']} | {r['Accuracy 95% CI']} | {r['Linkage AUC-ROC']} | {r['AUC 95% CI']} |\n")
        fp.write("\n")

    print("\n" + "="*85, flush=True)
    print("=== TABLE II: H2 FREQUENCY-DOMAIN PRIVACY LEAKAGE BREAKDOWN ===", flush=True)
    print("="*85, flush=True)
    print(f"{'Frequency Band':<28} | {'Accuracy (%)':<18} | {'AUC-ROC':<18}", flush=True)
    print("-" * 85, flush=True)
    for r in records:
        print(f"{r['Frequency Band']:<28} | {r['Linkage Accuracy (%)']:<6} {r['Accuracy 95% CI']:<10} | {r['Linkage AUC-ROC']:<6} {r['AUC 95% CI']:<10}", flush=True)
    print("="*85, flush=True)
    print(f"Saved results to {csv_path} and {md_path}", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate H2 frequency privacy leakage.")
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_h2_evaluation(num_pairs=args.num_pairs, seed=args.seed)
