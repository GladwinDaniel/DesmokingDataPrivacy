"""
Train Siamese Privacy Linkage Attacker
Part of Phase 1: Privacy Attack Infrastructure (H1)

Trains a Siamese ResNet-18 attacker on balanced pairs from the 14 training clusters.
Evaluates on the validation clusters.
Saves best checkpoint to checkpoints/privacy_attacker/best_linkage_attacker.pth.
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
from sklearn.metrics import roc_auc_score, accuracy_score

from models.privacy_attacker import SiameseLinkageAttacker
from data.linkage_dataset import SurgicalLinkageDataset



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
    acc = accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_targets, all_preds)
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
    acc = accuracy_score(all_targets, [1 if p >= 0.5 else 0 for p in all_preds])
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except Exception:
        auc = 0.5
    return avg_loss, acc, auc, all_preds, all_targets


def train_attacker(epochs=10,
                   batch_size=32,
                   lr=1e-4,
                   num_train_pairs=2000,
                   num_val_pairs=400,
                   output_dir="project_files/checkpoints/privacy_attacker",
                   seed=42):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Privacy Linkage Attacker on {device}...")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Datasets: train and val linkage pairs
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

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc
            }, best_ckpt)
            print(f"  -> Saved new best checkpoint with Val AUC: {val_auc:.4f}")

    print(f"\nTraining completed! Best Val AUC: {best_val_auc:.4f}. Checkpoint saved to {best_ckpt}")
    return best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Privacy Linkage Attacker")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_train_pairs", type=int, default=2000)
    parser.add_argument("--num_val_pairs", type=int, default=400)
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
