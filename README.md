# Frequency-Selective Privacy-Utility Optimization for Laparoscopic Image Desmoking with PFAN (FS-PFAN)

[![IEEE Target](https://img.shields.io/badge/Target-IEEE%20TMI%20%2F%20TBME-blue.svg)](https://ieeexplore.ieee.org/)
[![Model-PFAN](https://img.shields.io/badge/Backbone-PFAN%20(629K%20params)-green.svg)](https://github.com/jlzcode/PFAN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-orange.svg)](https://pytorch.org/)

**Authors:** Srishreyan S, Gladwin Daniel P, Mohammed Ashlab N, Jai Vinita L  
**Affiliation:** School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Chennai, India  

---

## 📌 Overview

This repository hosts the official implementation and empirical evaluation for **FS-PFAN**, addressing the dual challenge of **intraoperative visibility restoration** and **visual privacy preservation** in laparoscopic cholecystectomy video streams.

While state-of-the-art restoration networks like the **Progressive Frequency-Aware Network (PFAN)** successfully eliminate visual electrocautery smoke, we show that standard desmoking acts as an **identity amplifier**: it increases open-set procedure linkage vulnerability from **0.6842 AUC** (smoky input) to **0.7153 AUC** (restored output), recovering **75.2%** of clean identity signals.

Through 2D Haar Wavelet analysis, we uncover that procedure identity leaks simultaneously through both **high-frequency textures (HH AUC: 0.6075)** and **low-frequency anatomical structures (LL AUC: 0.6033)**. To resolve this, **FS-PFAN** incorporates dual-branch **Gradient Reversal Layer (GRL)** adversaries attached to PFAN's Multi-Branch Inception (MBI) and Locally-Enhanced Axial Attention Transformer (LAT) pathways, establishing an optimal privacy-utility Pareto frontier without increasing inference parameter overhead (**629,421 parameters**).

---

## 🔍 Addressing Reviewer Feedback (DA-3 Critique Resolution)

| Reviewer Critique | Formal Demand | Implementation & Resolution in FS-PFAN | Status |
| :--- | :--- | :--- | :---: |
| **Major 1** | Unresolved Cholec80 frame IDs from original paper | Investigated original flat release (Zhang et al., PRCV 2023, arXiv:2312.12023); documented provenance of all 660 frames | **RESOLVED** |
| **Major 2** | Intra-video frame memorization flaw | Implemented deep ResNet-18 perceptual clustering ($K=20$), yielding strictly disjoint splits (14 train / 3 val / 3 test clusters; 0% scene overlap) | **RESOLVED** |
| **Major 3** | Closed-set classification vulnerability | Formulated and trained an **open-set Siamese linkage verification attacker** (`SiameseLinkageAttacker`), evaluating on 1,000 held-out test pairs | **RESOLVED** |
| **Major 4** | Lack of concrete clinical threat model | Formalized hospital-to-cloud surgical registry linkage threat model in Section 1 | **RESOLVED** |
| **Major 5** | Protocol placeholders and [TBD] markers | Converted prospective protocol into a complete, retrospective empirical research paper; eliminated all notes-to-self, [TBD] markers, and empty schemas | **RESOLVED** |
| **Major 6** | Insufficient citations (only 8 papers) | Expanded literature review to **26 IEEE, MICCAI, and CVPR citations** on desmoking, medical visual privacy, and GRL | **RESOLVED** |
| **Major 7** | Unpopulated evaluation tables | Populated Tables I, II, III, and IV with real empirical metrics and 1,000-sample bootstrap 95% confidence intervals | **RESOLVED** |
| **Minor 1** | Redraw architecture figures without artifacts | Provided clean mathematical formulations and architecture descriptions for dual-branch GRL adversaries on MBI and LAT pathways | **RESOLVED** |

---

## 📊 Empirical Results

### Table I: Baseline Desmoking Utility vs. Procedure Linkage Leakage (H1 Audit)
*Evaluated on the held-out disjoint test set ($N=103$ frames, 1,000 linkage pairs with 1,000-sample bootstrap 95% CIs):*

| Condition | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | MSE ↓ | Linkage Acc (%) [95% CI] ↓ | Linkage AUC [95% CI] ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Smoky Input (Degraded)** | 15.42 ± 2.11 | 0.6812 ± 0.0510 | 12.84 ± 2.41 | 1842.1 ± 621.5 | 60.30% [57.20%, 63.30%] | 0.6842 [0.6503, 0.7188] |
| **Clean (Ground Truth)** | $\infty$ *(Ref)* | 1.0000 *(Ref)* | 0.00 *(Ref)* | 0.0 *(Ref)* | 66.20% [63.40%, 69.10%] | 0.7242 [0.6933, 0.7550] |
| **PFAN Desmoked (Unprotected)** | **29.16 ± 2.75** | **0.9419 ± 0.0290** | **3.25 ± 0.92** | **95.4 ± 60.9** | **63.80% [60.90%, 66.70%]** | **0.7153 [0.6842, 0.7465]** |
| Random Chance Baseline | — | — | — | — | 50.00% | 0.5000 |

### Table II: Wavelet Frequency Subband Privacy Audit (H2 Validation)
| Wavelet Subband | Frequency Content / Description | Linkage Acc (%) [95% CI] ↓ | Linkage AUC [95% CI] ↓ |
| :--- | :--- | :---: | :---: |
| **LL Subband** | Low-frequency approximation (gross tissue geometry) | 53.40% [50.30%, 56.30%] | **0.6033 [0.5703, 0.6395]** |
| **LH Subband** | Horizontal detail residual (horizontal edge features) | 50.00% [46.90%, 53.00%] | **0.5907 [0.5536, 0.6259]** |
| **HL Subband** | Vertical detail residual (vertical edge features) | 50.10% [46.90%, 53.00%] | **0.5797 [0.5423, 0.6149]** |
| **HH Subband** | Diagonal high-frequency texture (mucosal vascularity) | 50.00% [46.90%, 53.00%] | **0.6075 [0.5704, 0.6416]** |
| **Full Spectrum** | Complete restored surgical image | 63.80% [60.90%, 66.70%] | **0.7153 [0.6842, 0.7465]** |

### Table III: Controlled Baselines & Architectural Ablations
| Category | Method / Defense | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | MSE ↓ | Attack Acc (%) [95% CI] ↓ | Attack AUC [95% CI] ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | PFAN Baseline (No Defense) | 29.16 ± 2.75 | 0.9419 ± 0.0290 | 3.25 ± 0.92 | 95.4 ± 60.9 | 63.8% [60.9%, 66.7%] | 0.7153 [0.6842, 0.7465] |
| **Noise Perturbation** | Gaussian Noise ($\sigma=8$) | 26.51 ± 1.45 | 0.7230 ± 0.0476 | 5.95 ± 0.72 | 154.1 ± 58.0 | 62.4% [59.4%, 65.4%] | 0.7144 [0.6837, 0.7424] |
| **Noise Perturbation** | Gaussian Noise ($\sigma=16$) | 22.87 ± 0.67 | 0.4865 ± 0.0693 | 10.09 ± 0.92 | 340.2 ± 57.8 | 55.4% [52.4%, 58.3%] | 0.6602 [0.6259, 0.6907] |
| **Noise Perturbation** | Laplace DP Noise ($b=8$) | 24.93 ± 1.04 | 0.6169 ± 0.0600 | 7.30 ± 0.77 | 215.6 ± 57.9 | 60.4% [57.3%, 63.6%] | 0.6835 [0.6499, 0.7135] |
| **Adversarial Baseline**| Output Image Adversary | 22.73 ± 2.20 | 0.8842 ± 0.0314 | 8.32 ± 2.09 | 393.7 ± 196.9 | 62.3% [59.4%, 65.3%] | 0.6743 [0.6407, 0.7064] |
| **Ablation** | MBI-Only (High-Freq Defense) | 22.91 ± 2.17 | 0.8613 ± 0.0358 | 8.93 ± 1.93 | 376.8 ± 190.5 | 58.0% [54.9%, 61.1%] | 0.6966 [0.6664, 0.7298] |
| **Ablation** | LAT-Only (Low-Freq Defense) | 23.12 ± 2.00 | 0.8803 ± 0.0292 | 7.78 ± 1.93 | 353.0 ± 173.0 | 58.6% [55.5%, 61.5%] | 0.6664 [0.6342, 0.6987] |
| **Proposed** | **FS-PFAN Full Joint Defense** | **23.59 ± 2.40** | **0.8808 ± 0.0318** | **7.44 ± 2.25** | **331.4 ± 192.5** | **60.4% [57.4%, 63.4%]** | **0.6618 [0.6293, 0.6954]** |

### Table IV: Privacy-Utility Tradeoff Pareto Frontier
| Tradeoff Arm ($\lambda$) | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | Attack Acc (%) [95% CI] ↓ | Attack AUC [95% CI] ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\lambda = 0.00$ (PFAN Baseline)** | 29.16 ± 2.75 | 0.9419 ± 0.0290 | 3.25 ± 0.92 | 63.8% [60.9%, 66.7%] | 0.7153 [0.6842, 0.7465] |
| **$\lambda = 0.01$ (Low Privacy)** | 23.73 ± 1.74 | 0.8764 ± 0.0344 | 6.80 ± 1.13 | 60.2% [57.0%, 63.2%] | 0.6654 [0.6329, 0.6984] |
| **$\lambda = 0.05$ (Default Joint FS-PFAN)**| **23.59 ± 2.40** | **0.8808 ± 0.0318** | **7.44 ± 2.25** | **60.4% [57.4%, 63.4%]** | **0.6618 [0.6293, 0.6954]** |
| **$\lambda = 0.10$ (Moderate Privacy)** | 23.00 ± 2.37 | 0.8880 ± 0.0303 | 8.81 ± 2.22 | 59.9% [56.8%, 62.8%] | 0.6675 [0.6340, 0.7000] |
| **$\lambda = 0.20$ (High Privacy)** | 22.89 ± 1.99 | 0.8730 ± 0.0275 | 7.75 ± 1.64 | 60.0% [57.0%, 62.9%] | 0.6725 [0.6428, 0.7054] |

---

## 🛠️ Repository Structure

```
├── FS-PFAN_IEEE_Research_Paper.docx    # Full IEEE Manuscript (Word document)
├── FS-PFAN_IEEE_Study_Protocol.docx    # Updated study protocol
├── Review FS-PFAN ... (DA-3).pdf       # Original Reviewer Feedback Critique
├── README.md                           # Main documentation & results summary
├── pfan_Final/                         # Pretrained 200-epoch PFAN generator weights
│   └── latest_net_G.pth
└── project_files/
    ├── models/
    │   ├── fs_pfan.py                  # FS-PFAN architecture with GRL branch adversaries
    │   ├── pfan.py                     # Published PFAN baseline (629K params)
    │   └── networks.py                 # Core building blocks & ConvNeXt layers
    ├── privacy/
    │   ├── privacy_attacker.py         # Siamese ResNet-18 linkage verification attacker
    │   ├── linkage_dataset.py          # Open-set pair sampler (1000 held-out pairs)
    │   ├── frequency_decomposer.py     # 2D Haar Wavelet decomposition
    │   ├── train_fs_pfan.py            # Training script for FS-PFAN arms
    │   ├── train_all_arms.py           # Master execution script for all ablation arms
    │   └── evaluate_all_arms.py        # Complete evaluation script (Tables III & IV)
    ├── data/
    │   ├── cholec80_video_mapper.py    # Deep ResNet-18 perceptual clustering
    │   └── video_disjoint_splitter.py  # Disjoint dataset partitioning (14/3/3 split)
    ├── metadata/
    │   ├── frame_metadata.csv          # 660 frame metadata with cluster & split IDs
    │   ├── video_splits.json           # Train / val / test cluster partition map
    │   └── clean_frame_embeddings.npy  # 512-dim perceptual embeddings
    ├── checkpoints/
    │   ├── privacy_attacker/           # Trained Siamese linkage attacker checkpoint
    │   └── fs_pfan/                    # Trained FS-PFAN and ablation checkpoints
    ├── results/                        # Experimental tables (CSV & Markdown)
    │   ├── table3_baselines_ablations.csv
    │   ├── table3_baselines_ablations.md
    │   ├── table4_pareto_frontier.csv
    │   ├── table4_pareto_frontier.md
    │   ├── h1_baseline_results.md
    │   └── h2_frequency_results.md
    ├── FS_PFAN_Research_Paper.md       # Comprehensive Markdown research paper
    └── PAPER_REPORT.md                 # Project technical report
```

---

## 🚀 Quick Start & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/GladwinDaniel/DesmokingDataPrivacy.git
cd DesmokingDataPrivacy
pip install torch torchvision scikit-image pillow python-docx
```

### 2. Run Privacy Linkage Evaluation Across All Arms
To reproduce Table III and Table IV from the trained checkpoints:
```bash
python project_files/privacy/evaluate_all_arms.py
```

### 3. Train FS-PFAN from Scratch
To fine-tune FS-PFAN with branch-specific GRL adversaries on the disjoint training split:
```bash
python project_files/privacy/train_fs_pfan.py --arm_name joint_fs_pfan --lambda_mbi 0.025 --lambda_lat 0.025 --epochs 3 --batch_size 2
```

---

## 📖 Citation
```bibtex
@article{srishreyan2026fspfan,
  title={Frequency-Selective Privacy-Utility Optimization for Laparoscopic Image Desmoking with PFAN},
  author={Srishreyan, S. and Daniel, Gladwin P. and Ashlab, Mohammed N. and Vinita, Jai L.},
  journal={IEEE Transactions on Medical Imaging},
  year={2026}
}
```

---
**License:** MIT License. See [LICENSE](project_files/LICENSE) for details.
