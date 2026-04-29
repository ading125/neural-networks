# Adversarial Robustness and Calibration on CIFAR-10

## 1. Experimental Setup

This project studies the relationship between adversarial robustness and probabilistic calibration on CIFAR-10. The notebook uses the same dataset and preprocessing as the class adversarial-attacks notebook: CIFAR-10 images are converted to PyTorch tensors and kept in the `[0, 1]` pixel range. Keeping the images in this range makes the adversarial attacks easier to interpret, because FGSM and PGD perturbations can be clipped directly back to valid image values.

The experiment is split into two model-capacity settings. First, I train and evaluate a small CNN, `SmallCIFARCNN`. Then I repeat the same pipeline with a larger CNN, `LargeCIFARCNN`, which has more channels and an extra convolutional block. This lets me compare whether increasing model capacity changes the robustness-calibration relationship.

For each architecture, I run two training methods. The first is standard training with cross-entropy loss on clean CIFAR-10 images. The second is FGSM-based adversarial training. In adversarial training, each batch is first attacked with FGSM using the current model, and then the model is optimized on those adversarial examples using cross-entropy loss.

Each trained model is evaluated on clean test images and under two white-box attacks:

- **FGSM:** a one-step gradient-sign attack.
- **PGD:** a stronger iterative attack that repeatedly updates the input and projects it back into the allowed epsilon ball.

For each setting, I report clean accuracy, FGSM robust accuracy, PGD robust accuracy, negative log-likelihood (NLL), expected calibration error (ECE), and reliability diagrams. The notebook also plots ECE versus epsilon, clean accuracy versus epsilon, and robust accuracy under FGSM and PGD versus epsilon.

## 2. Quantitative Results

The following tables should be filled from the final output cells in the notebook after running it from top to bottom.

### Small CNN

| Training | Attack | Epsilon | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|---:|
| Standard | Clean | 0 | `[fill]` | `[fill]` | `[fill]` |
| Standard | FGSM | 8/255 | `[fill]` | `[fill]` | `[fill]` |
| Standard | PGD | 8/255 | `[fill]` | `[fill]` | `[fill]` |
| FGSM AT | Clean | 0 | `[fill]` | `[fill]` | `[fill]` |
| FGSM AT | FGSM | 8/255 | `[fill]` | `[fill]` | `[fill]` |
| FGSM AT | PGD | 8/255 | `[fill]` | `[fill]` | `[fill]` |

### Large CNN

| Training | Attack | Epsilon | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|---:|
| Standard | Clean | 0 | `[fill]` | `[fill]` | `[fill]` |
| Standard | FGSM | 8/255 | `[fill]` | `[fill]` | `[fill]` |
| Standard | PGD | 8/255 | `[fill]` | `[fill]` | `[fill]` |
| FGSM AT | Clean | 0 | `[fill]` | `[fill]` | `[fill]` |
| FGSM AT | FGSM | 8/255 | `[fill]` | `[fill]` | `[fill]` |
| FGSM AT | PGD | 8/255 | `[fill]` | `[fill]` | `[fill]` |

The epsilon-sweep plots provide the main trend information. For each model, the ECE-vs-epsilon plot shows how calibration changes as the perturbation budget grows. The clean-accuracy plot should stay mostly flat because clean inputs do not depend on epsilon. The robust-accuracy plot should show FGSM and PGD accuracy decreasing as epsilon increases, with PGD usually being the stronger attack.

## 3. Critical Discussion

### Does FGSM adversarial training improve robustness?

FGSM adversarial training should be evaluated separately against FGSM and PGD. If the FGSM robust accuracy of the adversarially trained model is higher than the standard model at the same epsilon, then FGSM adversarial training improved robustness against the attack it was trained on. This is the expected outcome because the model directly sees FGSM examples during training.

PGD is a stricter test. If PGD robust accuracy also improves, then adversarial training helped beyond the specific one-step attack. If FGSM accuracy improves but PGD accuracy does not, then the model may have learned robustness that is specific to FGSM rather than general robustness against stronger iterative attacks.

### Does increased robustness correlate with better or worse calibration?

The answer depends on whether ECE and NLL move in the same direction as robust accuracy. If adversarial training improves robust accuracy and lowers ECE/NLL, then increased robustness correlates with better calibration. If robust accuracy improves but ECE/NLL increases, then the model is more robust but less calibrated.

This comparison should be made separately for FGSM and PGD. A model can become better calibrated under FGSM while still being poorly calibrated under PGD. That distinction matters because PGD is a stronger attack and often reveals weaknesses that FGSM alone misses.

### Are adversarially trained models overconfident or underconfident?

The reliability diagrams answer this visually. A model is **overconfident** when its predicted confidence is higher than its empirical accuracy. In the diagram, this appears when the confidence curve is above the accuracy bars. A model is **underconfident** when its predicted confidence is lower than empirical accuracy.

For adversarial examples, overconfidence is especially important. If the model is wrong but still assigns high confidence, then the model is not only inaccurate; it is also unreliable. The reliability diagrams should be inspected for the clean, FGSM, and PGD settings for both the small and large CNNs.

### How do calibration metrics evolve as epsilon increases?

As epsilon increases, the attacks are allowed to make larger changes to the input image. The expected pattern is that robust accuracy decreases while ECE and NLL increase. This means the model becomes both less accurate and less well calibrated.

If the adversarially trained model has lower ECE than the standard model at larger epsilons, then adversarial training improves calibration under attack. If ECE rises even when robust accuracy improves, then adversarial training helps classification robustness but does not fully fix confidence reliability.

The larger CNN should be compared to the small CNN using the same metrics. If the larger model has better robust accuracy and lower ECE/NLL, then increased capacity improves both robustness and calibration. If it has higher accuracy but worse ECE, then capacity improves prediction performance without necessarily improving probability quality.

## 4. Conclusion

This project evaluates whether adversarial robustness and calibration improve together or trade off against one another on CIFAR-10. The notebook compares standard training and FGSM adversarial training for both a small CNN and a larger CNN. The key conclusions should be based on the final CIFAR-10 output tables and plots: whether FGSM adversarial training improves FGSM and PGD robustness, whether improved robustness corresponds to lower or higher ECE/NLL, and whether the larger CNN changes the robustness-calibration relationship.

Overall, the most important analysis is not only whether the model predicts correctly under attack, but whether its confidence remains trustworthy. A robust model that is still overconfident under PGD is not fully reliable, while a model with both higher robust accuracy and lower calibration error gives stronger evidence that robustness and calibration improved together.
