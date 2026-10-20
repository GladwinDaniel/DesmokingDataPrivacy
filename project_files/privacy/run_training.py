"""
Execute Siamese Linkage Attacker Training and Checkpoint Saving
Phase 1: Privacy Attack Infrastructure (H1)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from linkage_dataset import SurgicalLinkageDataset
from privacy_attacker import SiameseLinkageAttacker
from train_attacker import train_epoch, evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Siamese Privacy Attacker Training on {device} ===", flush=True)

    epochs = 6
    batch_size = 32
    lr = 1e-4

    torch.manual_seed(42)

    # 1000 train pairs across 14 train clusters
    print("Preparing training pairs (N=1000)...", flush=True)
    train_ds = SurgicalLinkageDataset(split="train", condition="clean", num_pairs=1000, seed=42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # 200 val pairs across 3 val clusters
    print("Preparing validation pairs (N=200)...", flush=True)
    val_ds = SurgicalLinkageDataset(split="val", condition="clean", num_pairs=200, seed=43)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("Initializing Siamese Linkage Attacker (ResNet-18 backbone)...", flush=True)
    model = SiameseLinkageAttacker(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = os.path.abspath(os.path.join(_this_dir, "..", "checkpoints", "privacy_attacker"))
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt = os.path.join(output_dir, "best_linkage_attacker.pth")

    best_val_auc = 0.0

    print("\n--- Training Progress ---", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%, AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, AUC: {val_auc:.4f}", flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc
            }, best_ckpt)
            print(f"  --> Saved new best model to {best_ckpt} (Val AUC: {val_auc:.4f})", flush=True)

    print(f"\nTraining Complete! Best Val AUC: {best_val_auc:.4f}", flush=True)
    print(f"Attacker checkpoint ready at: {best_ckpt}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("EXCEPTION CAUGHT IN MAIN:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

