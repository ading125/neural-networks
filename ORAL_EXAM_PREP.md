# ORAL_EXAM_PREP

This file is an exam survival guide for the repository, not a README. Everything below is grounded in the actual repo files, mainly [`adv_attacks_clean_gpu_v2 (1).ipynb`](</C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/neural-networks/adv_attacks_clean_gpu_v2 (1).ipynb>), with supporting context from [`README.md`](</C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/neural-networks/README.md>) and [`report.md`](</C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/neural-networks/report.md>).

If something is missing from code or outputs, it is labeled **unclear from repo**.

# 1. Project Overview

## What problem this project solves

This project studies a specific question in deep learning: **how adversarial robustness relates to probabilistic calibration**.

In plain language:

- Robustness asks: if we slightly perturb an input in a malicious way, does the model still classify it correctly?
- Calibration asks: when the model says "I am 90% confident," is it actually right about 90% of the time?

The project is not trying to invent a new architecture. It is trying to **measure a tradeoff**:

- Standard training usually gives better clean accuracy.
- Adversarial training may improve robustness.
- But the important research question is whether that robustness helps or hurts calibration.

## What dataset/task it uses

The dataset is **CIFAR-10**, loaded in notebook Part I with `torchvision.datasets.CIFAR10` inside [`adv_attacks_clean_gpu_v2 (1).ipynb`](</C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/neural-networks/adv_attacks_clean_gpu_v2 (1).ipynb>).

Task:

- 10-class image classification.
- Images are RGB, shape `3 x 32 x 32`.
- Classes are the standard CIFAR-10 categories such as airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.

## What the input and output are

Input:

- A batch of images shaped `[batch_size, 3, 32, 32]`.
- Pixel values are kept in `[0, 1]` because the code uses only `transforms.ToTensor()` and no normalization.

Output:

- Raw class scores, usually called **logits**, shaped `[batch_size, 10]`.
- These logits are turned into probabilities with `softmax` only when calibration metrics are computed in `compute_ece(...)` and `reliability_diagram_from_logits(...)`.

## What the final model is trying to learn

Each model is trying to learn a mapping:

- from image pixels
- to one of 10 CIFAR-10 classes

Under standard training, the model learns to minimize classification error on clean images.

Under adversarial training, the model learns something stricter:

- it must still classify correctly even after the input is deliberately perturbed by FGSM

More careful wording:

- the adversarial-training objective is trying to make predictions more stable in a local `L_infinity` neighborhood around each training example
- the repo results support improved robustness under the tested FGSM and PGD settings, but they do not directly prove a general smoothness property

## What the main experimental goal is

The main experimental goal is:

1. Compare **standard training** vs **FGSM adversarial training**.
2. Compare a **small CNN** vs a **larger CNN**.
3. Measure clean accuracy, adversarial accuracy, NLL, ECE, and reliability diagrams.
4. Answer whether robustness and calibration improve together or trade off.

The key experiment is implemented by:

- `SmallCIFARCNN`
- `LargeCIFARCNN`
- `train_one_epoch(...)`
- `train_one_epoch_adv_fgsm(...)`
- `fgsm_attack(...)`
- `pgd_attack(...)`
- `evaluate_model_over_eps(...)`

# 2. Repository Map

## `adv_attacks_clean_gpu_v2 (1).ipynb`

What it does:

- This is the real project.
- It contains the full pipeline: setup, data loading, model definitions, training, attack generation, evaluation, calibration metrics, plots, and result tables.

Why it exists:

- The repository is notebook-centric.
- Instead of splitting code into modules, the authors kept the whole experiment in one place for class submission and demonstration.

How it connects:

- All important logic lives here.
- `README.md` explains the notebook at a high level.
- `report.md` interprets the notebook's stored results.

Important functions/classes in the notebook:

- `set_seed(seed=0)`
- `train_one_epoch(...)`
- `eval_clean_accuracy(...)`
- `clamp01(...)`
- `fgsm_attack(...)`
- `pgd_attack(...)`
- `train_one_epoch_adv_fgsm(...)`
- `collect_logits_and_labels(...)`
- `compute_nll(...)`
- `compute_ece(...)`
- `metric_dict(...)`
- `reliability_diagram_from_logits(...)`
- `evaluate_model_over_eps(...)`
- `plot_epsilon_results(...)`
- `plot_reliability_for_eps(...)`
- `summarize_at_effect(...)`
- `SmallCIFARCNN`
- `LargeCIFARCNN`

## `README.md`

What it does:

- Gives the high-level project framing.
- States the research question and notebook structure.

Why it exists:

- It is the quick orientation file for someone opening the repo.

How it connects:

- It matches the notebook sections and confirms the intended deliverables.
- It is useful in the oral exam for explaining the motivation before going into code.

## `report.md`

What it does:

- Summarizes the final stored metrics and discusses the findings.

Why it exists:

- It acts like the written report built from notebook results.

How it connects:

- The numbers in `report.md` match the notebook outputs at `epsilon = 8/255`.
- It helps you explain conclusions, especially the clean-accuracy vs robustness tradeoff and calibration behavior under attack.

## `data/` folder

What it does:

- Stores downloaded CIFAR-10 files.

Why it exists:

- `datasets.CIFAR10(root="./data", download=True, ...)` downloads there automatically.

How it connects:

- The code depends on it at runtime, but the folder is not committed.
- Exact contents are **unclear from repo** because they are not tracked.

## `.gitignore`

What it does:

- Prevents local runtime artifacts from being committed.

Why it exists:

- Standard repo hygiene.

How it connects:

- It explicitly ignores `data/`, `.ipynb_checkpoints/`, `__pycache__/`, and `*.pyc`.
- That matches the notebook workflow, since CIFAR-10 is downloaded into `./data`.

# 3. End-to-End Pipeline

## Step 1: Data loading and preprocessing

In notebook Part I:

- `batch_size = 128`
- `transform = transforms.Compose([transforms.ToTensor()])`
- `train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)`
- `test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)`

Important design choice:

- There is **no mean/std normalization**.
- Images remain in `[0, 1]`.

Why that matters:

- Adversarial attacks like FGSM and PGD add perturbations directly to pixels.
- Because the valid image range is `[0, 1]`, the helper `clamp01(...)` can simply clip the attacked image back into the legal range.

DataLoader details:

- `shuffle=True` for training
- `shuffle=False` for testing
- `pin_memory=True` only if CUDA is available
- `num_workers=4` on non-Windows, else `0`, then capped by CPU count

## Step 2: Model construction

The notebook defines two architectures:

- `SmallCIFARCNN`
- `LargeCIFARCNN`

Both are plain CNN classifiers:

- convolution
- ReLU
- max pooling
- flatten
- fully connected layers

No batch normalization, dropout, residual connections, or attention are used.

## Step 3: Training loop

Standard training uses `train_one_epoch(model, loader, optimizer)`.

Code reference:

- notebook Part II, function `train_one_epoch(...)`

Per batch:

1. A batch `x` of shape `[B, 3, 32, 32]` and labels `y` of shape `[B]` is loaded.
2. `x` and `y` are moved to `device`.
3. `optimizer.zero_grad(set_to_none=True)` clears parameter gradients.
4. `logits = model(x)` produces shape `[B, 10]`.
5. `loss = F.cross_entropy(logits, y)` produces a scalar.
6. `loss.backward()` computes gradients with respect to model parameters.
7. `optimizer.step()` updates the parameters.
8. The code accumulates `loss.item() * x.size(0)` and divides by the total number of examples, so the returned epoch loss is example-weighted.

This is classic supervised learning with backpropagation.

Adversarial training uses `train_one_epoch_adv_fgsm(model, loader, optimizer, eps)`.

Code references:

- notebook Part II, function `train_one_epoch_adv_fgsm(...)`
- notebook Part II, function `fgsm_attack(...)`

Per batch:

1. Start from clean batch `x` and labels `y`.
2. Build attacked inputs using `x_adv = fgsm_attack(model, x, y, eps=eps)`.
3. Inside `fgsm_attack(...)`, the notebook:
   - sets `model.eval()`
   - creates `x_adv = x.clone().detach().requires_grad_(True)`
   - computes logits of shape `[B, 10]`
   - computes `F.cross_entropy(logits, y)`
   - gets the input gradient with `torch.autograd.grad(loss, x_adv)[0]`
   - adds `eps * sign(grad)` to the input
   - clips the result with `clamp01(...)`
4. Back in `train_one_epoch_adv_fgsm(...)`, the optimizer gradients are zeroed.
5. `logits_adv = model(x_adv)` is computed on the attacked batch.
6. `loss = F.cross_entropy(logits_adv, y)` is computed.
7. `loss.backward()` and `optimizer.step()` update the parameters.

So the training difference is simple but conceptually important:

- standard training optimizes on clean inputs
- adversarial training optimizes on FGSM-perturbed inputs generated from the current model

Important subtlety:

- `fgsm_attack(...)` calls `model.eval()`, and `train_one_epoch_adv_fgsm(...)` does **not** switch the model back to `train()` before the adversarial forward/update.
- In this repository that has almost no practical effect because the models have no dropout and no batch normalization.
- In a different architecture, this would matter and should be fixed.

Another precise point:

- The adversarial examples are detached before the training update.
- So the parameter update does **not** backpropagate through the attack-generation process itself. It only trains on the already-generated adversarial inputs.

## Step 4: Loss function

The loss is always **cross-entropy**, implemented with `F.cross_entropy(...)`.

Why it makes sense:

- This is a 10-class classification problem.
- Cross-entropy compares predicted class distribution to the one-hot true label.
- It strongly penalizes confident wrong predictions.

That last point is especially relevant because calibration is also being studied. A model that is confidently wrong under attack gets bad NLL and usually bad ECE.

## Step 5: Optimization

The optimizer is **Adam**:

- `torch.optim.Adam(..., lr=1e-3)`

Used for:

- `small_model`
- `small_adv_model`
- `large_model`
- `large_adv_model`

No scheduler is used. No weight decay is visible. No gradient clipping is visible.

## Step 6: Evaluation

Evaluation is driven by `evaluate_model_over_eps(...)`.

Code references:

- notebook Part II, functions `collect_logits_and_labels(...)`, `metric_dict(...)`, `compute_nll(...)`, `compute_ece(...)`, `evaluate_model_over_eps(...)`

It loops over:

- `eps_list = [0.0, 0.5/255, 1/255, 2/255, 4/255, 8/255]`

For each epsilon:

1. Evaluate clean metrics using `collect_logits_and_labels(..., attack_fn=None)`
2. If epsilon is not zero, evaluate FGSM metrics
3. Evaluate PGD metrics with:
   - `steps=10`
   - `alpha=eps/4`
   - `random_start=True`

Metrics produced by `metric_dict(...)`:

- accuracy
- NLL from `compute_nll(...)`
- ECE from `compute_ece(...)`

More precise evaluation flow:

- `collect_logits_and_labels(...)` iterates over the loader, optionally replaces each batch with attacked inputs using `attack_fn(x, y)`, then evaluates `model(x_eval)` under `torch.no_grad()`.
- It appends logits and labels to CPU lists and concatenates them at the end.
- So NLL and ECE are computed once on the full collected evaluation set, not batch-by-batch and then averaged.

ECE implementation details from `compute_ece(...)`:

- 15 equal-width confidence bins on `[0, 1]`
- confidence is the maximum softmax probability for each example
- prediction is `argmax`
- the last bin includes confidence exactly equal to `1.0`
- ECE is a weighted sum of `|mean_confidence - mean_accuracy|` over non-empty bins

Note:

- Clean evaluation is recomputed for every epsilon even though it does not change. This is computationally redundant, but harmless. It simplifies plotting.

## Step 7: Saving/loading outputs

There is **no real checkpointing system in the repo**.

What is present:

- stored notebook outputs
- printed epoch logs
- rendered pandas tables
- plots and reliability diagrams

What is missing:

- no `torch.save(...)`
- no `torch.load(...)`
- no saved `.pt` or `.pth` model files
- no experiment config file

So if asked how results are persisted, the honest answer is:

- the notebook stores outputs inline, but there is no reusable model serialization pipeline

# 4. Model Architecture

## Small model: `SmallCIFARCNN`

Defined in notebook Part III.A.

Code reference:

- notebook Part III.A, class `SmallCIFARCNN`

Layers:

1. `conv1 = nn.Conv2d(3, 64, 3, padding=1)`
2. `conv2 = nn.Conv2d(64, 128, 3, padding=1)`
3. `conv3 = nn.Conv2d(128, 128, 3, padding=1)`
4. `conv4 = nn.Conv2d(128, 256, 3, padding=1)`
5. `fc1 = nn.Linear(256 * 2 * 2, 256)`
6. `fc2 = nn.Linear(256, 10)`

Forward pass:

- Input: `[B, 3, 32, 32]`
- After `conv1` because kernel `3`, padding `1`, stride default `1`: `[B, 64, 32, 32]`
- After `ReLU`: `[B, 64, 32, 32]`
- After max pool: `[B, 64, 16, 16]`
- After `conv2 + ReLU`: `[B, 128, 16, 16]`
- After max pool: `[B, 128, 8, 8]`
- After `conv3 + ReLU`: `[B, 128, 8, 8]`
- After max pool: `[B, 128, 4, 4]`
- After `conv4 + ReLU`: `[B, 256, 4, 4]`
- After max pool: `[B, 256, 2, 2]`
- Flatten: `[B, 1024]`
- After `fc1 + ReLU`: `[B, 256]`
- After `fc2`: `[B, 10]`

Approximate parameter count:

- about **783,370**, derived from the layer sizes in `SmallCIFARCNN`

## Large model: `LargeCIFARCNN`

Defined in notebook Part IV.A.

Code reference:

- notebook Part IV.A, class `LargeCIFARCNN`

Layers:

1. `conv1 = nn.Conv2d(3, 96, 3, padding=1)`
2. `conv2 = nn.Conv2d(96, 192, 3, padding=1)`
3. `conv3 = nn.Conv2d(192, 256, 3, padding=1)`
4. `conv4 = nn.Conv2d(256, 384, 3, padding=1)`
5. `conv5 = nn.Conv2d(384, 384, 3, padding=1)`
6. `fc1 = nn.Linear(384 * 1 * 1, 512)`
7. `fc2 = nn.Linear(512, 10)`

Forward pass:

- Input: `[B, 3, 32, 32]`
- After `conv1 + ReLU`: `[B, 96, 32, 32]`
- After pool 1: `[B, 96, 16, 16]`
- After `conv2 + ReLU`: `[B, 192, 16, 16]`
- After pool 2: `[B, 192, 8, 8]`
- After `conv3 + ReLU`: `[B, 256, 8, 8]`
- After pool 3: `[B, 256, 4, 4]`
- After `conv4 + ReLU`: `[B, 384, 4, 4]`
- After pool 4: `[B, 384, 2, 2]`
- After `conv5 + ReLU`: `[B, 384, 2, 2]`
- After pool 5: `[B, 384, 1, 1]`
- Flatten: `[B, 384]`
- After `fc1 + ReLU`: `[B, 512]`
- Output logits: `[B, 10]`

Approximate parameter count:

- about **3,026,250**, derived from the layer sizes in `LargeCIFARCNN`

So the large model has roughly **3.9x more parameters** than the small one.

## Activation functions

Both architectures use:

- `F.relu(...)` after each convolution
- `F.relu(...)` after the first fully connected layer

No output activation is used inside the model because:

- `F.cross_entropy(...)` expects logits, not softmax probabilities

## Why this architecture was chosen

What is directly supported by the repo:

- The notebook explicitly runs a small-CNN experiment in Part III and a larger-CNN experiment in Part IV.
- [`README.md`](</C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/neural-networks/README.md>) states that the goal is to study the robustness-calibration tradeoff and compare a larger CNN against the smaller baseline.

This is a good oral-exam point:

- The architecture is intentionally simple so the effect of adversarial training is easier to interpret.

## What alternatives could have been used

Relevant alternatives:

- MLP: simpler, but worse inductive bias for images
- CNN with batch normalization: likely easier optimization and maybe better clean accuracy
- ResNet-style architecture: stronger baseline and more realistic modern classifier
- Wide CNN with dropout or weight decay
- TRADES or PGD adversarial training instead of FGSM adversarial training

## Strengths and weaknesses

Strengths:

- Simple and easy to explain
- Appropriate for CIFAR-10
- Convolutions exploit spatial locality
- Two model sizes let the project test capacity effects

Weaknesses:

- No batch normalization
- No regularization beyond adversarial training
- Only 5 epochs, so training may be underdeveloped
- FGSM training is weaker than stronger adversarial training methods
- The model family is basic compared with modern robust vision baselines

# 5. Training Details

Everything below comes from the notebook unless stated otherwise.

## Hyperparameters

- `batch_size = 128`
- `small_epochs = 5`
- `large_epochs = 5`
- `small_adv_epochs = 5`
- `large_adv_epochs = 5`
- `lr = 1e-3`
- `adv_train_eps = 8/255`
- `pgd_steps = 10`
- PGD step size `alpha = eps / 4`
- `eps_list = [0.0, 0.5/255, 1/255, 2/255, 4/255, 8/255]`
- `n_bins = 15` inside `compute_ece(...)`
- `max_eval_batches = None`

## Optimizer

- Adam

Implication:

- Adam converges quickly and is convenient for notebook experiments.
- It is a practical choice for a short training budget.
- But it may not be the best optimizer for final robust accuracy compared with carefully tuned SGD-based training.

## Learning rate

- `1e-3`

Implication:

- Standard default Adam learning rate.
- Good for quick experiments.
- Since there is no scheduler, the model trains with the same step size throughout.

## Batch size

- `128`

Implication:

- Reasonably efficient on GPU
- Large enough for stable gradients
- Small enough to fit CIFAR-10 training comfortably

## Epochs

- Only `5`

Implication:

- This is a short run.
- The project is more like a controlled course experiment than a fully optimized training campaign.
- If challenged on absolute performance, a fair answer is that results are meaningful for comparison, but likely not the best achievable with longer training.

## Loss function

- Cross-entropy for both standard and adversarial training

Implication:

- Same objective across settings keeps the comparison clean.
- The only main change is what inputs the model sees during training.

## Metrics

- Accuracy
- NLL
- ECE
- Reliability diagrams

What each metric implies:

- Accuracy measures correctness.
- NLL measures how bad the full predicted distribution is, especially punishing confident mistakes.
- ECE measures mismatch between confidence and observed correctness.
- Reliability diagrams visualize calibration bin by bin.

## Regularization

Explicit regularization visible in code:

- none for standard training
- adversarial training acts as a form of robustness-oriented regularization

What is absent:

- no dropout
- no batch norm
- no weight decay
- no data augmentation beyond `ToTensor()`

Implication:

- The clean standard model may fit the training data well but stay fragile.
- The adversarially trained model sacrifices clean performance to become more stable under perturbation.

# 6. Architectural Decisions and Tradeoffs

## Decision 1: CNN instead of MLP

What was chosen:

- Convolutional neural networks: `SmallCIFARCNN` and `LargeCIFARCNN`

Why it makes sense:

- CIFAR-10 is image data.
- CNNs exploit local spatial patterns and translation-related structure.

Tradeoff:

- Better image inductive bias than MLPs
- But still limited compared with stronger architectures like ResNet

What could go wrong:

- If the CNN is too shallow or too plain, it may cap clean accuracy and robustness.

What to try next:

- ResNet-18 or WideResNet baseline

## Decision 2: Compare small vs large CNN

What was chosen:

- A capacity study with `SmallCIFARCNN` vs `LargeCIFARCNN`

Why it makes sense:

- It tests whether more capacity helps robustness and calibration.

Tradeoff:

- Larger models can learn richer features
- But they are slower, heavier, and may still not solve adversarial fragility alone

What could go wrong:

- More parameters may increase overfitting risk or simply not help much without better training

What to try next:

- Longer training, normalization layers, stronger robust training

## Decision 3: Keep pixels in `[0, 1]` with no normalization

What was chosen:

- `transforms.ToTensor()` only

Why it makes sense:

- It keeps attack interpretation simple.
- `eps` is directly in pixel scale.
- `clamp01(...)` is easy and correct for this representation.

Tradeoff:

- Simpler attack implementation
- But not necessarily the best for model optimization

What could go wrong:

- Model training may be less standardized than common normalized CIFAR-10 baselines.

What to try next:

- Compare against normalized-input training with carefully adapted attack bounds

## Decision 4: FGSM adversarial training instead of stronger methods

What was chosen:

- `train_one_epoch_adv_fgsm(...)`

Why it makes sense:

- FGSM is easy to implement and fast enough for a course project.
- It directly connects training to the white-box attack idea.

Tradeoff:

- Much cheaper than multi-step adversarial training
- But weaker than PGD-based adversarial training or TRADES

What could go wrong:

- FGSM training can fail to provide strong robustness against stronger attacks.
- In some settings it can lead to gradient masking or unstable robust behavior.

What to try next:

- PGD adversarial training
- TRADES
- stronger attack evaluation such as more PGD steps or AutoAttack

## Decision 5: Use PGD-10 for evaluation

What was chosen:

- `pgd_attack(..., steps=10, alpha=eps/4, random_start=True)`

Why it makes sense:

- PGD is stronger than FGSM and is a standard robustness stress test.

Tradeoff:

- More credible than only using FGSM
- But still not the strongest possible evaluation

What could go wrong:

- If PGD settings are too weak, robustness may be overestimated.

What to try next:

- more PGD steps
- multiple random restarts
- AutoAttack

## Decision 6: Measure calibration but do not optimize it directly

What was chosen:

- `compute_ece(...)`, `compute_nll(...)`, and reliability diagrams are used only for evaluation

Why it makes sense:

- The project wants to observe the robustness-calibration relationship first.

Tradeoff:

- Clearer experiment interpretation
- But no direct calibration improvement method is tested

What could go wrong:

- A model could be robust yet poorly calibrated on clean data, which is exactly what happened here.

What to try next:

- temperature scaling
- label smoothing
- calibration-aware robust objectives

## Decision 7: Short training budget

What was chosen:

- 5 epochs for every experiment

Why it makes sense:

- Keeps runtime manageable, especially with attack evaluation and multiple models.

Tradeoff:

- Faster experimentation
- Less mature final accuracy and robustness

What could go wrong:

- The final numbers may reflect undertraining as well as design choices.

What to try next:

- more epochs
- learning-rate schedule
- checkpoint-based model selection

# 7. Results and Interpretation

## Stored training logs

From notebook outputs:

### Small CNN, standard training

- Epoch 1: loss `1.5992`, clean test acc `53.72%`
- Epoch 2: loss `1.1540`, clean test acc `62.74%`
- Epoch 3: loss `0.9316`, clean test acc `69.54%`
- Epoch 4: loss `0.7836`, clean test acc `71.49%`
- Epoch 5: loss `0.6812`, clean test acc `74.16%`

### Small CNN, FGSM adversarial training

- Epoch 1: loss `2.0951`, clean test acc `41.98%`
- Epoch 2: loss `1.8916`, clean test acc `47.72%`
- Epoch 3: loss `1.8175`, clean test acc `50.75%`
- Epoch 4: loss `1.7627`, clean test acc `54.26%`
- Epoch 5: loss `1.7216`, clean test acc `54.75%`

### Large CNN, standard training

- Epoch 1: loss `1.6783`, clean test acc `49.89%`
- Epoch 2: loss `1.1897`, clean test acc `61.10%`
- Epoch 3: loss `0.9500`, clean test acc `67.98%`
- Epoch 4: loss `0.7843`, clean test acc `71.37%`
- Epoch 5: loss `0.6517`, clean test acc `72.88%`

### Large CNN, FGSM adversarial training

- Epoch 1: loss `2.0973`, clean test acc `39.42%`
- Epoch 2: loss `1.9058`, clean test acc `46.10%`
- Epoch 3: loss `1.8273`, clean test acc `47.87%`
- Epoch 4: loss `1.7747`, clean test acc `50.57%`
- Epoch 5: loss `1.7290`, clean test acc `53.06%`

These logs already show the central tradeoff:

- standard training reaches higher clean accuracy
- adversarial training keeps loss higher and clean accuracy lower

## Final stored metrics at `epsilon = 8/255`

These values appear in the notebook tables and are also summarized in [`report.md`](</C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/neural-networks/report.md>).

### Small CNN

| Training | Attack | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|
| Standard | Clean | 0.7416 | 0.7590 | 0.0157 |
| Standard | FGSM | 0.0550 | 4.9455 | 0.6850 |
| Standard | PGD-10 | 0.0069 | 8.0789 | 0.9186 |
| FGSM AT | Clean | 0.5475 | 1.3270 | 0.1471 |
| FGSM AT | FGSM | 0.3421 | 1.7488 | 0.0306 |
| FGSM AT | PGD-10 | 0.3105 | 1.8275 | 0.0609 |

### Large CNN

| Training | Attack | Accuracy | NLL | ECE |
|---|---|---:|---:|---:|
| Standard | Clean | 0.7288 | 0.7953 | 0.0236 |
| Standard | FGSM | 0.0562 | 4.8573 | 0.6830 |
| Standard | PGD-10 | 0.0035 | 9.1360 | 0.9449 |
| FGSM AT | Clean | 0.5306 | 1.3367 | 0.1334 |
| FGSM AT | FGSM | 0.3511 | 1.7316 | 0.0196 |
| FGSM AT | PGD-10 | 0.3241 | 1.8101 | 0.0408 |

## What the metrics mean

Accuracy:

- Fraction of correct predictions

NLL:

- Lower is better
- Strongly punishes confident wrong predictions

ECE:

- Lower is better
- Measures calibration gap between predicted confidence and actual accuracy

Interpretation example:

- A model can have low accuracy and also huge ECE under attack, meaning it is not just wrong, but **confidently wrong**.

## Are the results good or bad?

Depends on the criterion.

For clean accuracy:

- Standard models are clearly better.
- Around `73%` to `74%` clean accuracy after only 5 epochs is reasonable for a simple notebook CNN, but not strong compared with well-trained CIFAR-10 baselines.

For adversarial robustness:

- Standard models are very poor at high epsilon.
- PGD-10 accuracy near zero means they are highly fragile.

For calibration under attack:

- Standard models are very bad at high epsilon.
- ECE around `0.92` or `0.94` under PGD-10 is extremely poor.

For adversarial training:

- It substantially improves robustness and attacked calibration.
- But it reduces clean accuracy and worsens clean calibration.

## What patterns appear

### Pattern 1: Standard models collapse under attack

Example:

- Small CNN standard PGD-10 accuracy at `8/255` is `0.0069`
- Large CNN standard PGD-10 accuracy at `8/255` is `0.0035`

This means the models almost completely fail under a stronger white-box attack.

### Pattern 2: Standard models become overconfident under attack

Evidence:

- Large standard PGD-10 ECE is `0.9449`
- Small standard PGD-10 ECE is `0.9186`

So the attacked model is not just wrong. It is badly miscalibrated.

Strict wording:

- High ECE and high NLL support the claim that confidence is badly misaligned with correctness.
- The word "overconfident" is also consistent with `report.md` and the notebook's intended reliability-diagram analysis, but the diagram bars themselves are not extracted numerically in the repo text outputs.

### Pattern 3: FGSM adversarial training sharply improves robustness

Example:

- Small PGD-10 accuracy improves from `0.0069` to `0.3105`
- Large PGD-10 accuracy improves from `0.0035` to `0.3241`

This is a very large practical gain.

### Pattern 4: FGSM adversarial training also improves attacked calibration

Example:

- Small PGD-10 ECE improves from `0.9186` to `0.0609`
- Large PGD-10 ECE improves from `0.9449` to `0.0408`

That is the central result of the repo.

### Pattern 5: Clean accuracy and clean calibration get worse after adversarial training

Example:

- Small clean accuracy: `0.7416 -> 0.5475`
- Large clean accuracy: `0.7288 -> 0.5306`
- Small clean ECE: `0.0157 -> 0.1471`
- Large clean ECE: `0.0236 -> 0.1334`

This is the clean-accuracy vs robustness tradeoff.

### Pattern 6: Larger capacity helps a little, not dramatically

At `8/255` after adversarial training:

- Large PGD-10 accuracy `0.3241`
- Small PGD-10 accuracy `0.3105`

So bigger capacity helps slightly, but the training method matters more than model size.

## What the results say about the research question

The repo supports a nuanced answer:

- On **clean data**, standard training gives better accuracy and calibration.
- On **adversarial data**, FGSM adversarial training gives much better robustness and much better calibration.

So robustness and calibration do **not** move together uniformly across all settings.

A better statement is:

- adversarial training improves calibration **under attack**
- but worsens calibration **on clean inputs**

## Limitations

Real limitations visible from repo:

- Only CIFAR-10
- Only 2 architectures
- Only 5 epochs
- Only FGSM adversarial training
- No validation split is visible
- No multiple random seeds are reported
- No saved checkpoints
- No statistical uncertainty or confidence intervals
- No stronger robust benchmark like AutoAttack

Professor-safe wording:

- The conclusions are meaningful within this experimental setup, but not broad enough to claim universal behavior across datasets, architectures, or training methods.

# 8. Likely Professor Questions

Below are more than 25 likely oral questions with strong answers.

## 1. What is the core question of this project?

The core question is whether improving adversarial robustness also changes calibration, and whether that change is beneficial or harmful. The repo studies this by comparing standard training and FGSM adversarial training on CIFAR-10 for `SmallCIFARCNN` and `LargeCIFARCNN`.

## 2. Why did you use CIFAR-10?

CIFAR-10 is a standard small image benchmark, so it is practical for repeated experiments in a notebook. It is complex enough to need CNNs, but small enough to support clean training, adversarial training, and multiple evaluation attacks in one project.

## 3. Why use a CNN instead of an MLP?

Because the input is image data. CNNs exploit spatial structure and local patterns, while an MLP would ignore that structure and usually perform worse on images like CIFAR-10.

## 4. Why compare two CNN sizes?

To isolate the effect of model capacity. The notebook uses `SmallCIFARCNN` and `LargeCIFARCNN` so we can ask whether a larger feature extractor changes the robustness-calibration tradeoff.

## 5. What exactly does the model output?

The model outputs logits of shape `[batch_size, 10]`. These are raw class scores. Softmax is only applied later when the code computes calibration quantities like confidence in `compute_ece(...)`.

## 6. Why is cross-entropy the right loss here?

Because this is multi-class classification with 10 labels. Cross-entropy encourages the correct class logit to be high relative to the others, and it also penalizes confident wrong predictions, which is relevant when discussing NLL and calibration.

## 7. How does backpropagation apply in this project?

Backpropagation is used in two places. First, during training, gradients of cross-entropy with respect to model parameters are used to update the network in `train_one_epoch(...)` and `train_one_epoch_adv_fgsm(...)`. Second, in adversarial attacks, gradients of the loss with respect to the input image are used in `fgsm_attack(...)` and `pgd_attack(...)` to construct perturbations.

## 8. What is FGSM doing mathematically?

FGSM takes the sign of the gradient of the loss with respect to the input, then moves the image by `eps * sign(grad)` in that direction. The idea is to increase the loss as much as possible in one step under an `L_infinity` constraint.

## 9. Why is PGD considered stronger than FGSM?

FGSM is one gradient step. PGD takes many small steps, projecting back into the allowed epsilon ball after each step. That iterative search usually finds harder adversarial examples.

## 10. Why clip images with `clamp01(...)`?

Because valid pixel values are represented in `[0, 1]`. After adding adversarial perturbations, the image could leave that legal range, so `clamp01(...)` restores it.

## 11. Why did you keep the images in `[0, 1]` instead of normalizing?

Because the attack code becomes simpler and more interpretable. Epsilon directly corresponds to pixel-scale perturbation, and clipping back into the legal range is straightforward.

## 12. What is ECE?

Expected Calibration Error measures the average gap between confidence and empirical accuracy across confidence bins. If a model predicts 90% confidence but is only correct 60% of the time in that bin, it is miscalibrated.

## 13. What is NLL, and why include it if you already have accuracy?

NLL captures the quality of the entire predicted distribution, not just whether the top class is correct. It strongly penalizes confident mistakes, so it is useful when evaluating attacked models that may be overconfident.

## 14. What is a reliability diagram?

It is a plot that compares predicted confidence to actual accuracy across bins. Perfect calibration would lie near the diagonal.

## 15. What does the standard model learn?

The standard model mainly learns discriminative features for clean CIFAR-10 images. But the results suggest it does not learn decision boundaries that are stable to small adversarial perturbations.

## 16. What changes when you do adversarial training?

Instead of optimizing on clean images, the model optimizes on FGSM-perturbed images generated from the current model. That pushes it to be more stable around the training points, which improves adversarial robustness.

## 17. Why does adversarial training hurt clean accuracy?

Because the optimization objective becomes harder. The model is no longer only trying to classify clean points well; it is also trying to be correct in a neighborhood around them. That typically shifts capacity from clean fitting toward robustness.

## 18. How do you know the standard model is not robust?

Because under `PGD-10` at `8/255`, accuracy falls almost to zero: `0.0069` for the small CNN and `0.0035` for the large CNN.

## 19. How do you know the adversarially trained model is more robust?

Because those PGD-10 accuracies rise to `0.3105` and `0.3241`. That is a large improvement under a stronger attack, not just under FGSM.

## 20. How do you know the attacked standard model is overconfident?

Because it has terrible ECE and NLL under attack. For example, the large standard model under PGD-10 at `8/255` has accuracy `0.0035` but ECE `0.9449`, meaning it is often confidently wrong.

## 21. Did larger model capacity solve the problem by itself?

No. The larger standard CNN still collapses under attack. Capacity alone does not fix adversarial fragility. Training method mattered much more.

## 22. Why is the larger model only slightly better after adversarial training?

The notebook shows modest gains from extra capacity, but the dominant factor is adversarial training itself. Since the training budget is short and the architecture is simple, the larger model may not have had enough setup to realize a much bigger advantage.

## 23. How do you know the model is not overfitting?

Strictly speaking, that is **unclear from repo** because there is no explicit validation split or train-vs-validation curve analysis. We only have training loss and test accuracy logs. So we can say the repo does not provide a strong overfitting diagnosis.

## 24. What hyperparameter would you change first with more time?

I would increase the number of epochs first. All models are trained for only 5 epochs, which is short for CIFAR-10. After that I would test stronger adversarial training and a learning-rate schedule.

## 25. Why evaluate several epsilon values instead of only `8/255`?

Because robustness is not a single number. The epsilon sweep shows how performance degrades as attack strength increases, which is more informative than only checking one point.

## 26. Why is clean accuracy repeated for every epsilon in the evaluation table?

Because `evaluate_model_over_eps(...)` recomputes clean metrics inside the epsilon loop. It is redundant computationally, but convenient for putting all results into one dataframe for plotting.

## 27. Why use `alpha = eps/4` in PGD?

It is a practical step-size heuristic: small enough to avoid overshooting, large enough to make progress in 10 steps. The exact choice is somewhat empirical.

## 28. Why use `random_start=True` in PGD?

Random start makes the attack stronger by not always beginning exactly at the clean point. It reduces the chance that the attack gets stuck in a weak local behavior near the original input.

## 29. Is FGSM adversarial training enough to claim strong robustness?

No. It improves robustness significantly in this repo, but it is still weaker than stronger methods like PGD adversarial training or TRADES. So the claim should be limited to this setup.

## 30. Why does clean ECE get worse after adversarial training?

The adversarially trained model appears less calibrated on natural images, likely because the training objective shifts from fitting the clean data distribution well to defending a neighborhood around each example.

## 31. What are the main dataset limitations?

CIFAR-10 is a small, curated benchmark dataset. It is useful for controlled experiments, but the repo does not test whether these conclusions hold under distribution shift, on larger datasets, or in real deployment settings.

## 32. If this were used in the real world, why would calibration matter?

Because confidence affects decision-making. In safety-critical settings, a confidently wrong model is more dangerous than an uncertain model. Calibration tells us whether confidence can be trusted.

## 33. What part of the code would you show first in a walkthrough?

I would show the high-level path:

1. dataset setup in Part I
2. attacks and metrics in Part II
3. `SmallCIFARCNN` or `LargeCIFARCNN`
4. standard training cell
5. adversarial training cell
6. `evaluate_model_over_eps(...)`

That sequence explains the whole experiment clearly.

## 34. What should each teammate absolutely be able to explain?

Everyone should be able to explain:

- what FGSM and PGD are
- why cross-entropy is used
- what ECE and NLL mean
- why adversarial training lowers clean accuracy
- what the main final result was

## 35. What is the strongest single conclusion from the repository?

In this repo, FGSM adversarial training substantially improves robustness and calibration under attack, but it reduces clean accuracy and worsens clean calibration.

## 36. Why is `fgsm_attack(...)` implemented with `torch.autograd.grad(...)` instead of `loss.backward()`?

Because the attack only needs the gradient with respect to the input tensor `x_adv`, not gradients accumulated on every model parameter. `torch.autograd.grad(...)` directly returns that input gradient.

## 37. Does `fgsm_attack(...)` run in training mode or evaluation mode, and why does that matter?

It explicitly sets `model.eval()`. In this repository that has little practical effect because the models contain no dropout and no batch normalization. In a model with those layers, attack generation and subsequent training behavior would depend on the mode.

## 38. Is the PGD implementation really projecting onto the `L_infinity` ball?

Yes. After each update, it computes `delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)` and then uses `clamp01(x_orig + delta)`. That enforces the `L_infinity` budget around the original image and then clips to the valid pixel range.

## 39. Why is FGSM also within the `L_infinity` budget even without an explicit projection step?

Because it starts at the clean image and adds exactly `eps * sign(grad)` once. Before clipping, each pixel changes by at most `eps` in absolute value. Clipping to `[0, 1]` can only reduce that perturbation, not increase it.

## 40. Is this adversarial training a true min-max robust optimization?

Only approximately. The notebook generates a single-step FGSM adversarial example and then minimizes cross-entropy on that attacked input. That is much cheaper than a stronger inner maximization like PGD adversarial training.

## 41. Why is `eval_clean_accuracy(...)` separate from `collect_logits_and_labels(...)`?

`eval_clean_accuracy(...)` is a lightweight helper used after each epoch to print test accuracy quickly. `collect_logits_and_labels(...)` is the heavier helper that stores full logits so the notebook can compute NLL, ECE, and reliability diagrams.

## 42. Could the evaluation still be underestimating attack strength?

Yes. The repo uses 10-step PGD with one step-size rule and the default single random start used inside the function. A stronger evaluation with more steps or more restarts could reduce the reported robust accuracies further.

## 43. Why is there no softmax layer in `SmallCIFARCNN` or `LargeCIFARCNN`?

Because `F.cross_entropy(...)` expects raw logits and internally handles the log-softmax behavior it needs. Adding softmax in the model would be unnecessary and numerically less clean.

## 44. What exact tensor shapes enter `F.cross_entropy(...)` during training?

The logits have shape `[B, 10]` and the labels have shape `[B]`, where each label is an integer class index in `0` through `9`.

## 45. Why is the lack of batch normalization important in this notebook?

Because batch normalization affects optimization and behaves differently in train and eval modes. Its absence makes the architecture simpler, and it is also the reason the `eval()` mode subtlety in adversarial training is mostly harmless here.

## Red-Flag Questions

These are the questions a professor may ask if they think you memorized the summary without understanding the code.

### Red flag 1: In `train_one_epoch_adv_fgsm(...)`, is the model definitely back in training mode before `optimizer.step()`?

No. `fgsm_attack(...)` sets `model.eval()`, and the outer function does not switch back to `model.train()` before the adversarial forward/update. In this repo that is mostly harmless because there is no dropout or batch norm, but it is still a real code subtlety.

### Red flag 2: Are gradients from the attack-generation step used directly to update model parameters?

Not directly. `fgsm_attack(...)` computes input gradients, returns a detached adversarial tensor, and then the training update comes from a new forward/backward pass on `x_adv`.

### Red flag 3: Does `compute_ece(...)` average batch ECE values?

No. The notebook first concatenates logits and labels for the full evaluation set using `collect_logits_and_labels(...)`, then computes ECE once on the concatenated tensors.

### Red flag 4: Why does `evaluate_model_over_eps(...)` recompute clean metrics at every epsilon?

Because the function is written as one loop that appends rows in a common dataframe format for all attack settings. It is redundant computationally, but it simplifies the table and plotting logic.

### Red flag 5: Can you prove the flatten size `256 * 2 * 2` in `SmallCIFARCNN`?

Yes. Four max-pooling layers with kernel size `2` halve the spatial dimensions each time: `32 -> 16 -> 8 -> 4 -> 2`. So the tensor before flattening is `[B, 256, 2, 2]`.

### Red flag 6: Can you prove the flatten size `384 * 1 * 1` in `LargeCIFARCNN`?

Yes. Five max-pooling layers halve the spatial dimensions `32 -> 16 -> 8 -> 4 -> 2 -> 1`. So the tensor before flattening is `[B, 384, 1, 1]`.

### Red flag 7: Are the clean metrics really functions of epsilon?

No. They are repeated in the table because of the evaluation-loop structure, not because clean accuracy or clean ECE should change with epsilon.

### Red flag 8: What exactly is "confidence" in `compute_ece(...)`?

It is the maximum softmax probability for each example. It is not the raw logit and not necessarily the probability assigned to the true class.

### Red flag 9: What would become wrong or risky if batch normalization were added without fixing the adversarial-training loop?

The `eval()` versus `train()` mismatch would become meaningful, because batch norm uses different behavior and running statistics in those modes.

### Red flag 10: Why can’t you claim the model is "well calibrated" in general just because attacked ECE is low here?

Because ECE is one summary statistic, it depends on binning, and these results are only for this dataset, these attacks, these architectures, and this training budget. Low ECE here means better calibration in this setup, not universal calibration quality.

# 9. "If You Get Stuck" Answers

Use these when you need an honest but competent fallback.

- "We chose this mainly because it gave us a clean, interpretable experiment rather than the strongest possible benchmark."
- "The main tradeoff is clean accuracy versus adversarial robustness."
- "The calibration result depends on whether we mean clean inputs or adversarial inputs."
- "From the repo, the effect is clear qualitatively even if the setup is not exhaustive."
- "That detail is unclear from repo, so I would not overclaim it."
- "With more time, we would test a longer training schedule and a stronger adversarial-training method."
- "The architecture is deliberately simple so the training-method effect is easier to isolate."
- "FGSM training is computationally cheaper, but the tradeoff is that it is not the strongest robustness method."
- "The limitation is that we only tested CIFAR-10 and only a small number of model/training settings."
- "The model learns features for classification, but adversarial training also encourages local stability around the data points."
- "ECE tells us whether the confidence is trustworthy, not just whether the prediction is correct."
- "A near-zero robust accuracy under PGD means the model is highly fragile even if clean accuracy is decent."
- "The larger model helped slightly, but the training objective mattered more than raw size."
- "The notebook stores the outputs, but the repo does not implement a formal checkpointing pipeline."
- "I would present this as a controlled course experiment, not as a final state-of-the-art robustness study."
- "A code-level subtlety is that `fgsm_attack(...)` leaves the model in eval mode, but in this repo that is mostly harmless because there is no dropout or batch norm."

# 10. 2-Minute Presentation Script

This project studies the relationship between adversarial robustness and calibration on CIFAR-10. In the main notebook, we train two CNNs, `SmallCIFARCNN` and `LargeCIFARCNN`, under two settings: standard clean training and FGSM adversarial training. The goal is to compare not only clean accuracy, but also how the models behave under adversarial attacks and whether their confidence remains trustworthy.

The pipeline is straightforward. We load CIFAR-10 with pixel values in `[0, 1]`, build the CNN, train it with cross-entropy and Adam, and then evaluate it on clean images plus FGSM and PGD adversarial examples. For evaluation, we compute accuracy, negative log-likelihood, expected calibration error, and reliability diagrams using helper functions like `evaluate_model_over_eps(...)`, `compute_nll(...)`, and `compute_ece(...)`.

The main result is that standard training gives better clean accuracy, around 73 to 74 percent for the standard CNNs after 5 epochs, but those models collapse under adversarial attack and become extremely overconfident. FGSM adversarial training lowers clean accuracy to about 53 to 55 percent, but it greatly improves robustness and also improves calibration under attack. So the key conclusion is that robustness and calibration do not behave the same way on clean versus adversarial inputs: adversarial training hurts clean performance, but helps attacked performance a lot.

# 11. Individual Teammate Study Guide

## Chunk 1: Data and preprocessing

This person should understand:

- how CIFAR-10 is loaded in Part I of [`adv_attacks_clean_gpu_v2 (1).ipynb`](</C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/neural-networks/adv_attacks_clean_gpu_v2 (1).ipynb>)
- why `transforms.ToTensor()` is the only preprocessing
- why images are kept in `[0, 1]`
- why `clamp01(...)` is needed for adversarial examples
- what `batch_size`, `pin_memory`, and `num_workers` do
- the fact that no augmentation or normalization is used

## Chunk 2: Model architecture

This person should understand:

- the layer-by-layer structure of `SmallCIFARCNN`
- the layer-by-layer structure of `LargeCIFARCNN`
- tensor shape changes through pooling
- why logits are returned instead of probabilities
- why CNNs are appropriate for CIFAR-10
- the parameter-size difference and what it implies

## Chunk 3: Training and evaluation

This person should understand:

- `train_one_epoch(...)`
- `train_one_epoch_adv_fgsm(...)`
- the exact difference between `loss.backward()` in training and `torch.autograd.grad(...)` in attack generation
- how Adam and cross-entropy are used
- how `fgsm_attack(...)` works
- how `pgd_attack(...)` works
- why `x_adv` is detached before the parameter update
- what `adv_train_eps = 8/255` means
- how `evaluate_model_over_eps(...)` constructs the result tables
- why clean metrics are recomputed for every epsilon

## Chunk 4: Results and limitations

This person should understand:

- the final accuracy/NLL/ECE numbers at `8/255`
- why standard models fail under attack
- why adversarial training improves attacked robustness
- why clean accuracy drops after adversarial training
- why clean ECE worsens but attacked ECE improves
- the main limitations: short training, only CIFAR-10, only FGSM AT, no checkpointing, no validation split

## Chunk 5: Code walkthrough

This person should understand:

- notebook organization by Parts I to IV
- where each helper function is defined
- where standard training begins for each architecture
- where adversarial training begins for each architecture
- how the stored outputs connect to `report.md`
- what the repo does not contain, especially missing modular code and saved model files

## Minimum overlap everyone should know

Even if the project is split, every teammate should still be able to explain:

- the overall research question
- standard vs adversarial training
- FGSM vs PGD
- what ECE and NLL measure
- the final conclusion of the experiment
