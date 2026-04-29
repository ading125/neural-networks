# Adversarial Robustness and Calibration in Deep Neural Networks

This report summarizes the UC3M Neural Networks project implemented in [adv_attacks_clean_gpu_v2 (1).ipynb](./adv_attacks_clean_gpu_v2%20(1).ipynb). The repository studies how adversarial robustness and probabilistic calibration interact in convolutional neural networks on MNIST and CIFAR-10. All numerical values below come from the latest notebook outputs stored in the repository.

## 1. Introduction

Neural networks can achieve high clean accuracy while remaining vulnerable to small adversarial perturbations. This project examines that weakness under a white-box threat model, where the attacker has access to the model and its gradients. The analysis focuses on two related questions: how accuracy changes under adversarial attack, and how trustworthy the model's predicted probabilities remain under the same conditions.

The notebook addresses three comparisons. First, it establishes a standard small-CNN baseline on MNIST and evaluates robustness and calibration. Second, it compares standard training with FGSM adversarial training. Third, it studies model scaling with a larger MNIST CNN and then extends the robustness workflow to CIFAR-10. The relevant robustness curves and reliability diagrams are stored inline in [adv_attacks_clean_gpu_v2 (1).ipynb](./adv_attacks_clean_gpu_v2%20(1).ipynb).

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

The robustness sweeps report `FGSM`, `PGD-10`, and `PGD-40`. The calibration summaries report clean, `FGSM (eps=0.15)`, and `PGD (eps=0.15, steps=20)` for MNIST.

Calibration is measured with:

- **NLL (Negative Log-Likelihood):** cross-entropy on logits and labels
- **ECE (Expected Calibration Error):** discrepancy between confidence and empirical accuracy across confidence bins
- **Reliability diagrams:** plots comparing confidence and accuracy bin by bin

## 3. Quantitative Results

### 3.1 MNIST: small CNN

After 3 epochs of standard training, the small CNN reaches **98.79%** clean test accuracy. Its robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.988 | 0.953 | 0.945 |
| 0.10 | 0.988 | 0.869 | 0.773 |
| 0.15 | 0.988 | 0.688 | 0.283 |
| 0.20 | 0.988 | 0.434 | 0.012 |
| 0.25 | 0.988 | 0.215 | 0.000 |

Its stored calibration summary at `eps=0.15` is:

| Setting | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| Clean | 0.9879 | 0.0357 | 0.0017 |
| FGSM (`eps=0.15`) | 0.6877 | 0.9887 | 0.1560 |
| PGD (`eps=0.15`, 20 steps) | 0.3095 | 2.5812 | 0.5325 |

The FGSM-adversarially trained small CNN reaches **97.47%** clean test accuracy. Its stored robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.975 | 0.900 | 0.759 |
| 0.10 | 0.975 | 0.953 | 0.512 |
| 0.15 | 0.975 | 0.963 | 0.209 |
| 0.20 | 0.975 | 0.953 | 0.044 |
| 0.25 | 0.975 | 0.904 | 0.005 |

Its stored calibration summary at `eps=0.15` is:

| Setting | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| Clean | 0.9747 | 0.0798 | 0.0033 |
| FGSM (`eps=0.15`) | 0.9631 | 0.1175 | 0.0081 |
| PGD (`eps=0.15`, 20 steps) | 0.2252 | 4.4144 | 0.6622 |

### 3.2 MNIST: calibration as epsilon increases

The following table shows how calibration evolves as epsilon increases for the small standard CNN and the small FGSM-adversarially trained CNN.

| Model | Epsilon | Attack | Accuracy | NLL | ECE |
|---|---:|---|---:|---:|---:|
| Small CNN (Standard) | 0.05 | FGSM | 0.9529 | 0.1362 | 0.0119 |
| Small CNN (Standard) | 0.05 | PGD | 0.9453 | 0.1636 | 0.0170 |
| Small CNN (Standard) | 0.10 | FGSM | 0.8688 | 0.4050 | 0.0477 |
| Small CNN (Standard) | 0.10 | PGD | 0.7779 | 0.7001 | 0.1073 |
| Small CNN (Standard) | 0.15 | FGSM | 0.6877 | 0.9887 | 0.1560 |
| Small CNN (Standard) | 0.15 | PGD | 0.3108 | 2.5803 | 0.5303 |
| Small CNN (Standard) | 0.20 | FGSM | 0.4341 | 1.9815 | 0.3585 |
| Small CNN (Standard) | 0.20 | PGD | 0.0208 | 6.3659 | 0.9317 |
| Small CNN (Standard) | 0.25 | FGSM | 0.2155 | 3.2447 | 0.5693 |
| Small CNN (Standard) | 0.25 | PGD | 0.0000 | 10.8123 | 0.9900 |
| Small CNN (FGSM AT) | 0.05 | FGSM | 0.8996 | 0.3060 | 0.0350 |
| Small CNN (FGSM AT) | 0.05 | PGD | 0.7595 | 0.8710 | 0.1381 |
| Small CNN (FGSM AT) | 0.10 | FGSM | 0.9529 | 0.1513 | 0.0116 |
| Small CNN (FGSM AT) | 0.10 | PGD | 0.5189 | 2.0253 | 0.3456 |
| Small CNN (FGSM AT) | 0.15 | FGSM | 0.9631 | 0.1175 | 0.0081 |
| Small CNN (FGSM AT) | 0.15 | PGD | 0.2247 | 4.4102 | 0.6627 |
| Small CNN (FGSM AT) | 0.20 | FGSM | 0.9533 | 0.1456 | 0.0054 |
| Small CNN (FGSM AT) | 0.20 | PGD | 0.0532 | 8.4219 | 0.8918 |
| Small CNN (FGSM AT) | 0.25 | FGSM | 0.9044 | 0.3168 | 0.0148 |
| Small CNN (FGSM AT) | 0.25 | PGD | 0.0082 | 12.9303 | 0.9625 |

### 3.3 MNIST: larger CNN

After 3 epochs of standard training, the larger CNN reaches **99.08%** clean test accuracy. Its robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.991 | 0.963 | 0.956 |
| 0.10 | 0.991 | 0.886 | 0.817 |
| 0.15 | 0.991 | 0.725 | 0.477 |
| 0.20 | 0.991 | 0.476 | 0.078 |
| 0.25 | 0.991 | 0.225 | 0.001 |

Its stored calibration summary at `eps=0.15` is:

| Setting | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| Clean | 0.9908 | 0.0288 | 0.0024 |
| FGSM (`eps=0.15`) | 0.7249 | 0.8968 | 0.1340 |
| PGD (`eps=0.15`, 20 steps) | 0.5041 | 1.7822 | 0.3348 |

The FGSM-adversarially trained large CNN reaches **99.04%** clean test accuracy. Its stored robustness sweep is:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.05 | 0.990 | 0.983 | 0.974 |
| 0.10 | 0.990 | 0.974 | 0.912 |
| 0.15 | 0.990 | 0.976 | 0.553 |
| 0.20 | 0.990 | 0.981 | 0.107 |
| 0.25 | 0.990 | 0.977 | 0.004 |

Its stored calibration summary at `eps=0.15` is:

| Setting | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| Clean | 0.9904 | 0.0276 | 0.0022 |
| FGSM (`eps=0.15`) | 0.9764 | 0.0643 | 0.0030 |
| PGD (`eps=0.15`, 20 steps) | 0.5833 | 1.3558 | 0.2766 |

### 3.4 CIFAR-10 extension

After 5 epochs, the standard CIFAR-10 CNN reaches **71.30%** clean test accuracy:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.5/255 | 0.713 | 0.627 | 0.622 |
| 1/255 | 0.713 | 0.548 | 0.529 |
| 2/255 | 0.713 | 0.404 | 0.349 |
| 4/255 | 0.713 | 0.215 | 0.109 |
| 8/255 | 0.713 | 0.053 | 0.003 |

The FGSM-adversarially trained CIFAR-10 model reaches **62.20%** clean test accuracy:

| Epsilon | Clean | FGSM | PGD-40 |
|---|---:|---:|---:|
| 0.5/255 | 0.622 | 0.595 | 0.595 |
| 1/255 | 0.622 | 0.572 | 0.571 |
| 2/255 | 0.622 | 0.522 | 0.519 |
| 4/255 | 0.622 | 0.435 | 0.415 |
| 8/255 | 0.622 | 0.292 | 0.228 |

Calibration metrics for CIFAR-10 are not included in the stored notebook outputs.

## 4. Discussion

### 4.1 Standard training vs FGSM adversarial training

The stored results show the expected tradeoff between clean accuracy and robustness. On MNIST, standard training gives slightly higher clean accuracy for the small model (`0.9879` versus `0.9747`), but FGSM adversarial training greatly improves FGSM robustness. At `eps=0.15`, FGSM accuracy increases from `0.688` to `0.963`; at `eps=0.20`, it increases from `0.434` to `0.953`.

The improvement against PGD is weaker and depends on the model size. For the small CNN at `eps=0.15`, PGD-40 accuracy decreases from `0.283` for standard training to `0.209` for FGSM adversarial training, so FGSM adversarial training does not improve strong PGD robustness in that setting. For the large CNN, however, PGD-40 accuracy at `eps=0.15` improves from `0.477` to `0.553`, so increased capacity helps the adversarially trained model resist PGD more effectively.

On CIFAR-10, adversarial training lowers clean accuracy (`0.622` versus `0.713`) but improves robustness at larger epsilons. At `8/255`, FGSM accuracy rises from `0.053` to `0.292`, and PGD-40 accuracy rises from `0.003` to `0.228`.

### 4.2 Robustness-calibration tradeoff

The calibration summaries show that stronger attacks generally worsen calibration. For the standard small CNN, clean ECE is very low (`0.0017`) and NLL is `0.0357`. Under FGSM at `eps=0.15`, ECE rises to `0.1560` and NLL to `0.9887`. Under PGD-20, ECE rises further to `0.5325` and NLL to `2.5812`.

FGSM adversarial training improves calibration under FGSM attack for both MNIST architectures. For the small CNN at `eps=0.15`, FGSM ECE drops from `0.1560` to `0.0081`, and FGSM NLL drops from `0.9887` to `0.1175`. For the large CNN, FGSM ECE drops from `0.1340` to `0.0030`, and FGSM NLL drops from `0.8968` to `0.0643`.

The PGD results are more mixed. For the small CNN, FGSM adversarial training worsens PGD calibration at `eps=0.15`: PGD-20 ECE rises from `0.5325` to `0.6622`, and NLL rises from `2.5812` to `4.4144`. For the large CNN, adversarial training improves PGD calibration: PGD-20 ECE drops from `0.3348` to `0.2766`, and NLL drops from `1.7822` to `1.3558`.

As epsilon increases, calibration usually degrades along with accuracy. This is especially clear under PGD. For the standard small CNN, PGD ECE increases from `0.0170` at `eps=0.05` to `0.9900` at `eps=0.25`. For the FGSM-adversarially trained small CNN, PGD ECE increases from `0.1381` at `eps=0.05` to `0.9625` at `eps=0.25`. This suggests that stronger perturbations make the model's confidence less trustworthy, even when the model is adversarially trained.

The reliability diagrams stored in the notebook support the same conclusion visually. Clean predictions are close to the perfect-calibration line, while attacked predictions, especially PGD predictions, show much larger confidence-accuracy gaps. In the high-epsilon PGD cases, the models are often overconfident: they still assign high confidence even when accuracy has collapsed.

### 4.3 Effect of model scaling

Model scaling improves the stored MNIST results. Under standard training, clean accuracy increases from `0.9879` for the small CNN to `0.9908` for the larger model. At `eps=0.15`, FGSM accuracy improves from `0.6877` to `0.7249`, and PGD-20 accuracy improves from `0.3095` to `0.5041`.

Scaling also improves the adversarially trained model. The large FGSM-adversarially trained CNN reaches clean accuracy `0.9904`, FGSM accuracy `0.9764`, and PGD-20 accuracy `0.5833` at `eps=0.15`. This is better than the small FGSM-adversarially trained CNN, which reaches clean accuracy `0.9747`, FGSM accuracy `0.9631`, and PGD-20 accuracy `0.2252`.

Calibration follows a similar pattern. The large adversarially trained CNN has lower PGD-20 ECE (`0.2766`) and NLL (`1.3558`) than the small adversarially trained CNN (`0.6622` ECE and `4.4144` NLL). Within these stored outputs, larger capacity improves both robustness and calibration, although neither model is fully robust at high epsilon.

## 5. Conclusion

The project provides a notebook-based study of adversarial robustness and calibration. Standard training yields high clean accuracy on MNIST and lower clean accuracy on CIFAR-10, but robustness declines quickly as perturbations grow, especially under PGD. Calibration also worsens under attack, which shows that adversarial vulnerability is not only an accuracy problem: the predicted probabilities become less reliable too.

FGSM adversarial training improves robustness against FGSM very clearly. Its effect on PGD is more limited: the small MNIST CNN does not gain strong PGD robustness, while the larger CNN benefits more. Calibration behaves similarly. FGSM adversarial training greatly improves FGSM calibration, but PGD calibration can still be poor, especially for the smaller model. Overall, the repository supports a consistent conclusion: adversarial robustness and calibration are closely linked, PGD reveals weaknesses that FGSM alone can miss, and increased model capacity helps but does not completely solve the robustness-calibration problem.
