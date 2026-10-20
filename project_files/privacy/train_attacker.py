"""
Train Siamese Privacy Linkage Attacker
Phase 1: Privacy Attack Infrastructure (H1)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import argparse
import numpy as np
import torch
import torch.nn as nn
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


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        x1 = batch["img1"].to(device)
        x2 = batch["img2"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs.tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = fast_acc(all_targets, all_preds)
    auc = fast_auc(all_targets, all_preds)
    return avg_loss, acc, auc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x1 = batch["img1"].to(device)
            x2 = batch["img2"].to(device)
            labels = batch["label"].to(device)

            logits = model(x1, x2)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = fast_acc(all_targets, all_preds)
    auc = fast_auc(all_targets, all_preds)
    return avg_loss, acc, auc, all_preds, all_targets


def train_attacker(epochs=8,
                   batch_size=32,
                   lr=1e-4,
                   num_train_pairs=1000,
                   num_val_pairs=200,
                   output_dir=None,
                   seed=42):
    if output_dir is None:
        output_dir = os.path.join(_this_dir, "..", "checkpoints", "privacy_attacker")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Privacy Linkage Attacker on {device}...", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        train_ds = SurgicalLinkageDataset(split="train", condition="clean", num_pairs=num_train_pairs, seed=seed)
        val_ds = SurgicalLinkageDataset(split="val", condition="clean", num_pairs=num_val_pairs, seed=seed + 1)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = SiameseLinkageAttacker(pretrained=True).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_auc = 0.0
        best_ckpt = os.path.join(output_dir, "best_linkage_attacker.pth")

        print(f"Dataset sizes: Train pairs={len(train_ds)}, Val pairs={len(val_ds)}", flush=True)
        print("\n--- Starting Training ---", flush=True)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, AUC: {val_auc:.4f}", flush=True)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc
                }, best_ckpt)
                print(f"  -> Saved new best checkpoint with Val AUC: {val_auc:.4f}", flush=True)

        print(f"\nTraining completed! Best Val AUC: {best_val_auc:.4f}. Checkpoint saved to {best_ckpt}", flush=True)
        return best_ckpt
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Privacy Linkage Attacker")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_train_pairs", type=int, default=1000)
    parser.add_argument("--num_val_pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_attacker(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_train_pairs=args.num_train_pairs,
        num_val_pairs=args.num_val_pairs,
        seed=args.seed
    )
