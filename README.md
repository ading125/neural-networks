# Adversarial Robustness and Calibration in Deep Neural Networks

This project studies the relationship between adversarial robustness and probabilistic calibration in deep neural networks. The main notebook in the repository starts from standard clean training, adds FGSM-based adversarial training, evaluates robustness with FGSM and PGD attacks, and measures how model confidence behaves under both clean and adversarial conditions.

The primary artifact is:

- [adv_attacks_clean_gpu_v2 (1).ipynb](</C:/Users/Andrew/Desktop/Code Saves/Neural Networks/adv_attacks_clean_gpu_v2 (1).ipynb>)

## Project Goal

The goal is to understand the robustness-calibration tradeoff and how it changes as model capacity increases.

The notebook is organized around the following questions:

1. How well does a small CNN perform under standard clean training?
2. How much does robustness degrade under FGSM and PGD attacks?
3. How calibrated are the model's predicted probabilities on clean and adversarial inputs?
4. What changes when the model is trained with FGSM-based adversarial training?
5. How might the robustness-calibration relationship evolve as network size grows?

## What Is Included

The notebook currently includes:

- Standard training of a small CNN with cross-entropy loss on MNIST
- FGSM attack implementation
- PGD attack implementation
- Accuracy sweeps across multiple epsilon values
- FGSM adversarial training
- Side-by-side standard vs adversarially trained comparisons
- A CIFAR-10 extension of the same workflow
- Calibration analysis for the standard MNIST model:
  - Clean test accuracy
  - FGSM robust accuracy
  - PGD robust accuracy
  - Negative Log-Likelihood (NLL)
  - Expected Calibration Error (ECE)
  - Reliability diagrams

## Standard Training Section

For the standard small CNN, the notebook now evaluates the following:

- Clean test accuracy
- Robust accuracy under FGSM
- Robust accuracy under PGD
- Expected Calibration Error (ECE)
- Negative Log-Likelihood (NLL)
- Reliability diagrams for clean, FGSM, and PGD predictions

This section is intended to serve as the baseline before comparing against adversarial training and larger-capacity models.

## Main Methods

### Standard Training

The baseline model is trained on clean data using:

- Cross-Entropy loss
- Adam optimizer

### FGSM Adversarial Training

The adversarially trained model uses a mixed objective:

\[
\mathcal{L}_{mix} = (1 - \lambda)\,\mathcal{L}(f_\theta(x), y) + \lambda\,\mathcal{L}(f_\theta(x_{adv}), y)
\]

where `x_adv` is generated with FGSM using the current model.

### Robustness Evaluation

Robustness is evaluated with:

- FGSM: fast single-step attack
- PGD: stronger iterative attack in an \(L_\infty\) ball

### Calibration Evaluation

Calibration is measured with:

- NLL: penalizes confident incorrect predictions
- ECE: compares model confidence to empirical accuracy across confidence bins
- Reliability diagrams: visualize whether predicted confidence matches observed accuracy

## Repository Structure

```text
.
|-- adv_attacks_clean_gpu_v2 (1).ipynb
`-- README.md
```

## How To Run

This project is notebook-based.

1. Open the notebook in Jupyter or VS Code.
2. Run the cells in order from top to bottom.
3. Use `py` if you need to launch Python from the terminal on this machine.
4. Let the MNIST standard-training section finish before running robustness and calibration cells.
5. Run the adversarial training and CIFAR-10 sections afterward if you want the full comparison workflow.

## Recommended Execution Order

For the clearest report, use this order:

1. Setup and data loading
2. Train the standard MNIST small CNN
3. Evaluate clean, FGSM, and PGD robustness
4. Run the calibration section for the standard model
5. Train the FGSM adversarially trained model
6. Compare standard vs adversarial training
7. Repeat the workflow for CIFAR-10
8. Extend the notebook with larger-capacity models if your project requires scaling experiments

## Suggested Discussion Points

When writing up results, it is useful to comment on:

- Whether higher robustness reduces clean accuracy
- Whether stronger attacks worsen calibration
- Whether adversarial training improves or hurts confidence calibration
- Whether PGD reveals weaknesses that FGSM misses
- Whether larger models shift the tradeoff between robustness and calibration

## Notes

- The notebook keeps image values in the `[0, 1]` range so clipping and projection are easy to interpret.
- The MNIST section is the cleanest place to build the core report figures first.
- The CIFAR-10 section is harder and may require more epochs for stronger final performance.
- The code now includes function-level overview comments and detailed inline comments to make the experimental flow easier to follow.

## Next Extensions

Natural next steps for this project are:

- Add explicit large-model variants for the capacity-scaling study
- Compare calibration before and after adversarial training at multiple capacities
- Save summary tables and plots for use in a written report
- Add temperature scaling as a post-hoc calibration baseline
