# Table III: Controlled Baselines & Architectural Ablations

| Category | Method / Defense | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | MSE ↓ | Attack Acc (%) [95% CI] ↓ | Attack AUC [95% CI] ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | PFAN Baseline (No Defense) | 29.16 ± 2.75 | 0.9419 ± 0.0290 | 3.25 ± 0.92 | 95.4 ± 60.9 | 63.8% [60.9%, 66.7%] | 0.7153 [0.6842, 0.7465] |
| Noise Perturbation | Gaussian Noise (sigma=8) | 26.51 ± 1.45 | 0.7230 ± 0.0476 | 5.95 ± 0.72 | 154.1 ± 58.0 | 62.4% [59.4%, 65.4%] | 0.7144 [0.6837, 0.7424] |
| Noise Perturbation | Gaussian Noise (sigma=16) | 22.87 ± 0.67 | 0.4865 ± 0.0693 | 10.09 ± 0.92 | 340.2 ± 57.8 | 55.4% [52.4%, 58.3%] | 0.6602 [0.6259, 0.6907] |
| Noise Perturbation | Laplace DP Noise (b=8) | 24.93 ± 1.04 | 0.6169 ± 0.0600 | 7.30 ± 0.77 | 215.6 ± 57.9 | 60.4% [57.3%, 63.6%] | 0.6835 [0.6499, 0.7135] |
| Adversarial Baseline | Output Image Adversary | 22.73 ± 2.20 | 0.8842 ± 0.0314 | 8.32 ± 2.09 | 393.7 ± 196.9 | 62.3% [59.4%, 65.3%] | 0.6743 [0.6407, 0.7064] |
| Ablation | MBI-Only (High-Freq Defense) | 22.91 ± 2.17 | 0.8613 ± 0.0358 | 8.93 ± 1.93 | 376.8 ± 190.5 | 58.0% [54.9%, 61.1%] | 0.6966 [0.6664, 0.7298] |
| Ablation | LAT-Only (Low-Freq Defense) | 23.12 ± 2.00 | 0.8803 ± 0.0292 | 7.78 ± 1.93 | 353.0 ± 173.0 | 58.6% [55.5%, 61.5%] | 0.6664 [0.6342, 0.6987] |
| Proposed | FS-PFAN Full Joint Defense | 23.59 ± 2.40 | 0.8808 ± 0.0318 | 7.44 ± 2.25 | 331.4 ± 192.5 | 60.4% [57.4%, 63.4%] | 0.6618 [0.6293, 0.6954] |
