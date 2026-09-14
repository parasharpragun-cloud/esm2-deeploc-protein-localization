# ESM-2 + DeepLoc: Debugging Protein Subcellular Localization

A controlled reconstruction of an earlier ESM-2 + DeepLoc pipeline that had silently failed, producing F1 = 0.00 and accuracy = 0.00 on validation. Instead of tuning hyperparameters and hoping for the best, this project rebuilds the pipeline step by step to find out exactly what broke and why.

## Problem

Protein sequence → ESM-2 → sequence representation → classification head → localization prediction.

DeepLoc is a multi-label task: a protein can live in more than one part of the cell, so the model predicts 10 independent localization labels (Cytoplasm, Nucleus, Extracellular, Cell membrane, Mitochondrion, Plastid, Endoplasmic reticulum, Lysosome/Vacuole, Golgi apparatus, Peroxisome).

The original run of this pipeline returned zero on every metric. Rather than assume the model or the task was broken, the goal here was to isolate the cause.

## Approach

Every stage of the pipeline was verified independently before touching the model:

- Data loading and label extraction (`bloyal/deeploc` on Hugging Face)
- Tokenization and its interaction with ESM-2's sequence length limit
- ESM-2 forward pass and the shape of its output representations
- Classifier architecture and loss function
- Evaluation logic (sigmoid thresholding, F1, exact-match accuracy)

Once each piece was confirmed to work in isolation, a frozen ESM-2 baseline was compared against a fine-tuned ESM-2 model on the same data.

**Model:** `facebook/esm2_t6_8M_UR50D` (320-dim hidden size, 6 layers)
**Data subset:** 1,000 training proteins, 200 validation proteins
**Classifier:** single linear layer, 320 → 10
**Loss:** BCEWithLogitsLoss (independent binary decision per label)
**Threshold:** sigmoid output, 0.5 cutoff for a positive prediction

## What Was Actually Wrong

**1. Sequence length was never constrained to what ESM-2 supports.**
ESM-2 has `max_position_embeddings = 1026`, but the tokenizer's default `model_max_length` doesn't enforce that. Using `truncation=True` without an explicit `max_length` let some sequences tokenize to over 5,000 tokens, far past what the model can handle. In the 1,000-protein training subset, 112 proteins (11.2%) exceeded 1026 tokens; in validation, 24 of 200 (12%) did. Fixing this meant explicitly setting `max_length=1026`, which also resolved the GPU memory issues that came with it.

**2. A frozen ESM-2 backbone wasn't enough.**
With ESM-2 fully frozen and only the classification head trained, the model barely predicted any positives at the 0.5 threshold:

- Micro-F1: 0.0227
- Exact-match accuracy: 0.01

**3. Fine-tuning ESM-2 fixed it.**
Unfreezing ESM-2 and training it jointly with the classifier produced a large jump:

- Micro-F1: 0.5558
- Exact-match accuracy: 0.26

**4. What turned out not to be broken.**
The classifier dimensions (320 → 10), the multi-label formulation, label tensor shapes, prediction/label alignment, sigmoid usage, and the loss function were all correct from the start. The original zero score was not a tensor-shape or architecture bug, it was a sequence-length bug compounded by an under-adapted frozen representation.

## Results

| Experiment | Micro-F1 | Exact-match accuracy |
|---|---|---|
| Frozen ESM-2 baseline | 0.0227 | 0.01 |
| Fine-tuned ESM-2 | 0.5558 | 0.26 |

Per-class F1 (fine-tuned model):

| Class | F1 |
|---|---|
| Cytoplasm | 0.4901 |
| Nucleus | 0.7105 |
| Extracellular | 0.8485 |
| Cell membrane | 0.5405 |
| Mitochondrion | 0.6667 |
| Plastid | 0.0 |
| Endoplasmic reticulum | 0.0 |
| Lysosome/Vacuole | 0.0 |
| Golgi apparatus | 0.0 |
| Peroxisome | 0.0 |

## Why the Rare Classes Still Fail

The training subset is heavily imbalanced: Cytoplasm and Nucleus have 300+ examples each, while Plastid (30), Golgi apparatus (40), and Peroxisome (9) barely have enough to learn from. After fine-tuning, the model made zero positive predictions for any of the five rarest classes, which is what drives their F1 to zero. This is a data limitation of the controlled subset, not a failure of the model or the pipeline.

## Interpretation

ESM-2 converts an amino acid sequence into a 320-dimensional representation that captures patterns learned from large protein databases. The classification head then asks 10 independent biological questions of that representation: how likely is this protein to be found in each location. Because localization categories aren't mutually exclusive, this has to be a multi-label problem rather than single-choice classification.

The results show that ESM-2's pretrained representations do carry localization-relevant signal, but that signal only becomes usable once the model is allowed to adapt to the specific task. A frozen backbone with a linear head on top wasn't enough; fine-tuning was.

## Next Steps

This reconstruction gives a validated, working pipeline. The next stage is scaling it to the full DeepLoc dataset (rather than the 1,000/200 debugging subset) and evaluating generalization on the held-out test set, which should also address the rare-class collapse seen here.

## Stack

Python, PyTorch, Hugging Face `transformers` (ESM-2), Hugging Face `datasets`, scikit-learn (F1, accuracy), Google Colab (T4 GPU)
