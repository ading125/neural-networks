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

The following tables summarize the final stored notebook outputs at the largest attack budget, `epsilon = 8/255`.

### Small CNN

| Training | Attack | Epsilon | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|---:|
| Standard | Clean | 0 | 0.7416 | 0.7590 | 0.0157 |
| Standard | FGSM | 8/255 | 0.0550 | 4.9455 | 0.6850 |
| Standard | PGD-10 | 8/255 | 0.0069 | 8.0789 | 0.9186 |
| FGSM AT | Clean | 0 | 0.5475 | 1.3270 | 0.1471 |
| FGSM AT | FGSM | 8/255 | 0.3421 | 1.7488 | 0.0306 |
| FGSM AT | PGD-10 | 8/255 | 0.3105 | 1.8275 | 0.0609 |

### Large CNN

| Training | Attack | Epsilon | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|---:|
| Standard | Clean | 0 | 0.7288 | 0.7953 | 0.0236 |
| Standard | FGSM | 8/255 | 0.0562 | 4.8573 | 0.6830 |
| Standard | PGD-10 | 8/255 | 0.0035 | 9.1360 | 0.9449 |
| FGSM AT | Clean | 0 | 0.5306 | 1.3367 | 0.1334 |
| FGSM AT | FGSM | 8/255 | 0.3511 | 1.7316 | 0.0196 |
| FGSM AT | PGD-10 | 8/255 | 0.3241 | 1.8101 | 0.0408 |

The epsilon-sweep plots show the main trend. For standard training, increasing epsilon sharply lowers robust accuracy and raises ECE/NLL. For the small standard CNN, PGD-10 ECE rises from `0.0672` at `0.5/255` to `0.9186` at `8/255`; for the large standard CNN, PGD-10 ECE rises from `0.1007` to `0.9449`. FGSM adversarial training changes this pattern: clean accuracy drops, but attacked ECE and NLL are much lower at high epsilon.

## 3. Critical Discussion

### Does FGSM adversarial training improve robustness?

FGSM adversarial training improves robustness against both FGSM and PGD at `epsilon = 8/255` for both architectures. For the small CNN, FGSM accuracy increases from `0.0550` to `0.3421`, and PGD-10 accuracy increases from `0.0069` to `0.3105`. For the large CNN, FGSM accuracy increases from `0.0562` to `0.3511`, and PGD-10 accuracy increases from `0.0035` to `0.3241`.

This is important because PGD is a stronger attack than FGSM. The fact that PGD-10 accuracy also improves suggests that FGSM adversarial training gives some robustness beyond the exact one-step attack used during training. However, the clean accuracy is lower after adversarial training: the small CNN drops from `0.7416` to `0.5475`, and the large CNN drops from `0.7288` to `0.5306`. This shows the usual clean-accuracy versus robustness tradeoff.

### Does increased robustness correlate with better or worse calibration?

For adversarial inputs, increased robustness correlates with better calibration in these results. At `8/255`, the small CNN's FGSM ECE decreases from `0.6850` to `0.0306` after adversarial training, and PGD-10 ECE decreases from `0.9186` to `0.0609`. NLL also improves strongly: FGSM NLL drops from `4.9455` to `1.7488`, and PGD-10 NLL drops from `8.0789` to `1.8275`.

The same pattern appears for the large CNN. At `8/255`, FGSM ECE decreases from `0.6830` to `0.0196`, and PGD-10 ECE decreases from `0.9449` to `0.0408`. NLL also drops from `4.8573` to `1.7316` under FGSM and from `9.1360` to `1.8101` under PGD-10. Therefore, in this experiment, the adversarially trained models are both more robust and better calibrated on adversarial examples.

The clean-data calibration moves in the opposite direction. The small CNN's clean ECE increases from `0.0157` to `0.1471`, and the large CNN's clean ECE increases from `0.0236` to `0.1334`. So adversarial training improves calibration under attack but worsens calibration on clean inputs.

### Are adversarially trained models overconfident or underconfident?

The reliability diagrams show that the standard models become strongly overconfident under attack. This is also visible from the high ECE and NLL values: under PGD-10 at `8/255`, the standard small CNN has only `0.0069` accuracy but ECE `0.9186`, and the standard large CNN has only `0.0035` accuracy but ECE `0.9449`. These models are often confidently wrong.

The FGSM-adversarially trained models are much less overconfident on adversarial examples. Their PGD-10 ECE values at `8/255` are `0.0609` for the small CNN and `0.0408` for the large CNN. The reliability diagrams should therefore appear closer to the diagonal for adversarial inputs. On clean inputs, however, the adversarially trained models have worse ECE, which suggests that they are less well calibrated on natural images than the standard models.

### How do calibration metrics evolve as epsilon increases?

For standard training, calibration worsens as epsilon increases. The small standard CNN's FGSM ECE rises from `0.0629` at `0.5/255` to `0.6850` at `8/255`, while PGD-10 ECE rises from `0.0672` to `0.9186`. The large standard CNN follows the same trend, with PGD-10 ECE increasing from `0.1007` to `0.9449`.

For FGSM adversarial training, the trend is different. As epsilon increases, robust accuracy still decreases, but ECE does not explode in the same way. For the small adversarially trained CNN, PGD-10 ECE is `0.1333` at `0.5/255` and `0.0609` at `8/255`; for the large adversarially trained CNN, PGD-10 ECE is `0.1235` at `0.5/255` and `0.0408` at `8/255`. In these runs, adversarial training keeps calibration error much lower under large perturbations.

The larger CNN is slightly better than the small CNN after adversarial training at `8/255`: PGD-10 accuracy improves from `0.3105` to `0.3241`, and PGD-10 ECE improves from `0.0609` to `0.0408`. The improvement is modest, but it suggests that increased capacity helps the adversarially trained model slightly.

## 4. Conclusion

This project evaluates whether adversarial robustness and calibration improve together or trade off against one another on CIFAR-10. In the stored results, FGSM adversarial training improves both FGSM and PGD-10 robustness at `8/255` for the small and large CNNs. It also greatly improves calibration on adversarial examples, lowering both ECE and NLL under FGSM and PGD.

The main tradeoff is that clean accuracy and clean calibration become worse after adversarial training. Standard models are better on clean data, but they become highly overconfident under strong attacks. Adversarially trained models are less accurate on clean data, but they are much more robust and much better calibrated under adversarial perturbations. The larger CNN gives a small additional improvement under adversarial training, especially for PGD-10 robustness and ECE.
