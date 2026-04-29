# Adversarial Robustness and Calibration in Deep Neural Networks

This project studies the relationship between adversarial robustness and probabilistic calibration in deep neural networks using CIFAR-10. The main notebook starts from standard clean training, adds FGSM-based adversarial training, evaluates robustness with FGSM and PGD attacks, and measures how model confidence behaves under clean and adversarial conditions.

The primary artifact is:

- [adv_attacks_clean_gpu_v2 (1).ipynb](</C:/Users/Andrew/Desktop/Code Saves/Neural Networks/adv_attacks_clean_gpu_v2 (1).ipynb>)

The notebook is the combined end-to-end project file. It contains the PyTorch model implementations, training code, attack code, evaluation code, plots, and written explanations in one place.

## Project Goal

The goal is to understand the robustness-calibration tradeoff and how it changes when using FGSM adversarial training and a larger CNN architecture.

The notebook is organized around these questions:

1. How well does a small CNN perform on CIFAR-10 under standard training?
2. How much does robustness degrade under FGSM and PGD attacks?
3. How calibrated are the model's predicted probabilities on clean and adversarial inputs?
4. What changes when the same architecture is trained with FGSM adversarial training?
5. How does the robustness-calibration relationship change when using a larger CNN?

## What Is Included

The notebook includes:

- CIFAR-10 loading and preprocessing with images kept in `[0, 1]`
- PyTorch implementations of `SmallCIFARCNN` and `LargeCIFARCNN`
- Standard training with cross-entropy loss
- FGSM adversarial training
- FGSM and PGD attack implementations
- Clean, FGSM, and PGD accuracy evaluation
- Calibration metrics:
  - Expected Calibration Error (ECE)
  - Negative Log-Likelihood (NLL)
  - Reliability diagrams
- Plots for:
  - ECE vs epsilon
  - clean accuracy vs epsilon
  - robust accuracy under FGSM and PGD vs epsilon

## Notebook Roadmap

The notebook is divided into clearly labeled parts:

- `Part I: CIFAR-10 Dataset and Preprocessing`
  - loads CIFAR-10 from `./data`
  - keeps images in the `[0, 1]` range
- `Part II: Shared Training, Attacks, and Calibration Helpers`
  - defines shared helper functions used by both model sizes
  - includes training, attacks, metrics, and plotting helpers
- `Part III: Small CNN Experiments`
  - defines the small CIFAR-10 CNN
  - trains the small CNN with standard cross-entropy training
  - evaluates clean accuracy, FGSM/PGD robustness, ECE, NLL, and reliability diagrams
  - trains the same small CNN with FGSM adversarial training
  - repeats the same evaluation pipeline
- `Part IV: Larger CNN Experiments`
  - defines the larger CIFAR-10 CNN
  - repeats the full standard-training and FGSM-adversarial-training workflow
  - compares how scaling model capacity changes robustness and calibration

## Main Methods

### Standard Training

The baseline model is trained on clean CIFAR-10 images using:

- Cross-Entropy loss
- Adam optimizer

### FGSM Adversarial Training

The adversarially trained model generates FGSM adversarial examples at each training step and optimizes on those adversarial images.

### Robustness Evaluation

Robustness is evaluated with:

- FGSM: fast single-step attack
- PGD: stronger iterative attack in an \(L_\infty\) ball

### Calibration Evaluation

Calibration is measured with:

- NLL: penalizes confident incorrect predictions
- ECE: compares model confidence to empirical accuracy across confidence bins
- Reliability diagrams: visualize whether predicted confidence matches observed accuracy

## Deliverables

This repository covers the requested deliverables inside the notebook:

- PyTorch implementation of all models
  - `SmallCIFARCNN`
  - `LargeCIFARCNN`
- Training and evaluation code
  - standard training
  - FGSM adversarial training
  - clean, FGSM, and PGD evaluation
  - accuracy, NLL, ECE, and reliability diagrams

## Repository Structure

```text
.
|-- adv_attacks_clean_gpu_v2 (1).ipynb
|-- README.md
`-- report.md
```

The local `data/` folder is used by `torchvision.datasets` but is ignored by Git because datasets are downloaded automatically and are too large to push to GitHub.

## How To Run

1. Open the notebook in Jupyter or VS Code.
2. Run the cells in order from top to bottom.
3. Allow CIFAR-10 to download into `./data` if it is not already present.
4. After the notebook finishes, use the final tables and plots to update the 2-3 page report.

## Suggested Discussion Points

When writing up results, comment on:

- Whether FGSM adversarial training improves robustness against FGSM
- Whether FGSM adversarial training improves robustness against PGD
- Whether improved robustness comes with better or worse calibration
- Whether adversarially trained models look overconfident or underconfident in reliability diagrams
- How ECE and NLL evolve as epsilon increases
- Whether the larger CNN changes the robustness-calibration tradeoff
