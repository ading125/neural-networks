# Adversarial Robustness and Calibration in Deep Neural Networks

This report summarizes the UC3M Neural Networks project implemented in [adv_attacks_clean_gpu_v2 (1).ipynb](./adv_attacks_clean_gpu_v2%20(1).ipynb). The repository studies how adversarial robustness and probabilistic calibration interact in convolutional neural networks on MNIST and CIFAR-10. All numerical values below come directly from notebook outputs stored in the repository. When a metric was not saved, it is reported as not available.

## 1. Introduction

Neural networks can achieve high clean accuracy while remaining vulnerable to small adversarial perturbations. This project examines that weakness under a white-box threat model, where the attacker has access to the model and its gradients. The analysis focuses on two related questions: how accuracy changes under adversarial attack, and how trustworthy the model's predicted probabilities remain under the same conditions.

The notebook addresses three comparisons. First, it establishes a standard small-CNN baseline on MNIST and evaluates robustness and calibration. Second, it compares standard training with FGSM adversarial training. Third, it studies model scaling with a larger MNIST CNN and then extends the same robustness workflow to CIFAR-10. The relevant robustness curves and reliability diagrams are stored inline in [adv_attacks_clean_gpu_v2 (1).ipynb](./adv_attacks_clean_gpu_v2%20(1).ipynb); there are no standalone saved plot files in the repository.

## 2. Experimental Setup

The notebook uses MNIST and CIFAR-10, with images converted to tensors and kept in the `[0,1]` range. Training is implemented in PyTorch and configured to use one GPU if available.

Three models are defined:

- `SmallCNN` for MNIST: conv layers `1->32`, `32->64`, then fully connected layers `64*7*7->128->10`
- `LargeCNN` for MNIST: conv layers `1->64`, `64->128`, `128->256`, then fully connected layers `256*3*3->256->10`
- `SmallCIFARCNN` for CIFAR-10: conv layers `3->64->128->128->256`, then fully connected layers `256*2*2->256->10`

Standard training uses cross-entropy loss with Adam (`lr=1e-3`). The small and large MNIST models are trained for 3 epochs, and the CIFAR-10 model for 5 epochs.

The notebook also implements FGSM adversarial training. For each batch, adversarial examples are generated with FGSM against the current model, and the optimization step uses a mixed objective:

\[
\mathcal{L}_{mix} = (1-\lambda)\,\mathcal{L}(f_\theta(x), y) + \lambda\,\mathcal{L}(f_\theta(x_{adv}), y)
\]

For MNIST, adversarial training uses `eps=0.20`, `lam=0.5`, and 3 epochs. For CIFAR-10, it uses `eps=8/255`, `lam=0.5`, and 5 epochs.

Robustness is evaluated with two white-box `L_\infty` attacks:

- **FGSM:** a one-step gradient-sign attack
- **PGD:** an iterative projected gradient attack

The stored sweeps report `FGSM`, `PGD-10`, and `PGD-40`. An earlier MNIST section and the calibration summaries also report `PGD (eps=0.15, steps=20)`.

Calibration is measured with:

- **NLL (Negative Log-Likelihood):** cross-entropy on logits and labels
- **ECE (Expected Calibration Error):** discrepancy between confidence and empirical accuracy across confidence bins
- **Reliability diagrams:** plots comparing confidence and accuracy bin by bin

## 3. Quantitative Results

### 3.1 MNIST: small CNN

After 3 epochs of standard training, the small CNN reaches **98.84%** clean test accuracy. Its robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.988 | 0.954 | 0.945 |
| 0.10 | 0.988 | 0.869 | 0.770 |
| 0.15 | 0.988 | 0.685 | 0.269 |
| 0.20 | 0.988 | 0.421 | 0.009 |
| 0.25 | 0.988 | 0.204 | 0.000 |

Its stored calibration summary at `eps=0.15` is:

| Setting | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| Clean | 0.9884 | 0.0348 | 0.0023 |
| FGSM (`eps=0.15`) | 0.6848 | 0.9958 | 0.1550 |
| PGD (`eps=0.15`, 20 steps) | 0.2977 | 2.6129 | 0.5432 |

The FGSM-adversarially trained small CNN reaches **98.10%** clean test accuracy. Its stored robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.981 | 0.893 | 0.813 |
| 0.10 | 0.981 | 0.861 | 0.467 |
| 0.15 | 0.981 | 0.862 | 0.107 |
| 0.20 | 0.981 | 0.802 | 0.003 |
| 0.25 | 0.981 | 0.628 | 0.000 |

ECE and NLL for the adversarially trained small model are not available in the stored outputs.

### 3.2 MNIST: larger CNN

After 3 epochs of standard training, the larger CNN reaches **99.27%** clean test accuracy. Its robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.993 | 0.971 | 0.966 |
| 0.10 | 0.993 | 0.903 | 0.848 |
| 0.15 | 0.993 | 0.751 | 0.520 |
| 0.20 | 0.993 | 0.525 | 0.090 |
| 0.25 | 0.993 | 0.289 | 0.001 |

Its stored calibration summary at `eps=0.15` is:

| Setting | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| Clean | 0.9927 | 0.0227 | 0.0017 |
| FGSM (`eps=0.15`) | 0.7506 | 0.8034 | 0.1305 |
| PGD (`eps=0.15`, 20 steps) | 0.5434 | 1.6123 | 0.3074 |

The FGSM-adversarially trained large CNN reaches **98.92%** clean test accuracy. Its stored robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.989 | 0.973 | 0.878 |
| 0.10 | 0.989 | 0.981 | 0.370 |
| 0.15 | 0.989 | 0.996 | 0.002 |
| 0.20 | 0.989 | 0.997 | 0.000 |
| 0.25 | 0.989 | 0.993 | 0.000 |

ECE and NLL for the adversarially trained large model are not available in the stored outputs.

### 3.3 CIFAR-10 extension

After 5 epochs, the standard CIFAR-10 CNN reaches **73.45%** clean test accuracy:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.5/255 | 0.735 | 0.649 | 0.644 |
| 1/255 | 0.735 | 0.567 | 0.551 |
| 2/255 | 0.735 | 0.420 | 0.362 |
| 4/255 | 0.735 | 0.217 | 0.115 |
| 8/255 | 0.735 | 0.067 | 0.006 |

The FGSM-adversarially trained CIFAR-10 model reaches **57.53%** clean test accuracy:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.5/255 | 0.575 | 0.557 | 0.556 |
| 1/255 | 0.575 | 0.536 | 0.534 |
| 2/255 | 0.575 | 0.492 | 0.487 |
| 4/255 | 0.575 | 0.426 | 0.405 |
| 8/255 | 0.575 | 0.309 | 0.250 |

Calibration metrics for CIFAR-10 are not available in the stored outputs.

## 4. Discussion

### 4.1 Standard training vs FGSM adversarial training

The stored results show the expected tradeoff between clean accuracy and robustness. On MNIST, standard training gives the better clean result for the small model (`0.9884` versus `0.981`), but FGSM adversarial training improves FGSM robustness at larger perturbations. At `eps=0.20`, FGSM accuracy increases from `0.421` to `0.802`. On CIFAR-10, adversarial training lowers clean accuracy (`0.575` versus `0.735`) but improves robustness at larger epsilons; at `8/255`, FGSM accuracy rises from `0.067` to `0.309`, and PGD-40 from `0.006` to `0.250`.

These outcomes are consistent with the design of FGSM adversarial training: the model learns from both clean and adversarial samples, so it typically sacrifices some clean-data fit in exchange for improved resistance to gradient-based perturbations. However, the MNIST results also show that FGSM adversarial training does not guarantee strong PGD robustness. For the small model at `eps=0.15`, FGSM accuracy improves from `0.685` to `0.862`, but PGD-40 remains low at `0.107`.

### 4.2 Robustness-calibration tradeoff

The calibration summaries for the standard MNIST models show a clear tradeoff under attack. On clean data, both models are well calibrated: the small standard model has ECE `0.0023` and NLL `0.0348`, while the large standard model has ECE `0.0017` and NLL `0.0227`. Under FGSM at `eps=0.15`, both metrics worsen substantially. For the small model, ECE rises to `0.1550` and NLL to `0.9958`; under PGD-20, they rise further to `0.5432` and `2.6129`. The same pattern holds for the large model, though less severely: FGSM ECE `0.1305`, NLL `0.8034`; PGD-20 ECE `0.3074`, NLL `1.6123`.

The implication is that stronger attacks reduce not only accuracy but also the reliability of model confidence. In the stored MNIST summaries, PGD is especially informative because it produces both lower accuracy and worse calibration than FGSM. The corresponding reliability diagrams are stored inline in [adv_attacks_clean_gpu_v2 (1).ipynb](./adv_attacks_clean_gpu_v2%20(1).ipynb).

The repository does not store ECE or NLL for adversarially trained models, so a direct numerical calibration comparison between standard and adversarial training is not available.

### 4.3 Effect of model scaling

Model scaling improves the stored MNIST standard-model results. Clean accuracy increases from `0.9884` for the small CNN to `0.9927` for the larger model. At `eps=0.15`, FGSM accuracy improves from `0.6848` to `0.7506`, and PGD-20 accuracy from `0.2977` to `0.5434`. Calibration also improves: clean ECE decreases from `0.0023` to `0.0017`, and PGD-20 ECE from `0.5432` to `0.3074`.

Within the scope of the stored MNIST standard-model outputs, higher capacity is therefore associated with better clean accuracy, better adversarial accuracy, and lower ECE/NLL. Even so, the larger model still suffers substantial degradation under strong attack, so scaling helps but does not remove the underlying vulnerability.

## 5. Conclusion

The project provides a coherent notebook-based study of adversarial robustness and calibration. Standard training yields high clean accuracy on MNIST and lower clean accuracy on CIFAR-10, but robustness declines quickly as perturbations grow, especially under PGD. The calibration analysis shows that adversarial vulnerability is not only an accuracy problem: under attack, predicted confidence becomes much less reliable, as reflected in higher ECE and NLL and in the reliability diagrams stored in the notebook.

FGSM adversarial training improves robustness against FGSM and, on CIFAR-10, improves stored large-epsilon robustness relative to standard training, though at the cost of lower clean accuracy. The MNIST scaling experiment shows that a larger model can improve clean accuracy, adversarial accuracy, and calibration in the stored standard-model results. Overall, the repository supports a consistent conclusion: adversarial robustness and calibration are closely linked, PGD reveals weaknesses that FGSM alone can miss, and larger capacity helps but does not solve the robustness-calibration problem.
