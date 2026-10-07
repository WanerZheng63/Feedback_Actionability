# Classifying Actionability in Educational Feedback

Fine-tuned encoder models as a privacy-preserving alternative to LLMs for assessing the actionability of tutor feedback.

This repository contains the code and synthetic training data for:
> **Classifying Actionability in Educational Feedback: Fine-Tuned Encoder Models as a Privacy-Preserving Alternative to LLMs**

The trained model weights are available from the [latest release](https://github.com/wanna666/Feedback_Actionability/releases/tag/v1.0).
Please contact me on wanerzheng@hotmail.com if you have any questions.

---

## What this is

Feedback only helps students when it is **actionable** — when it tells them something specific they can do. This project trains compact encoder models to classify tutor feedback into three levels of actionability, and benchmarks them against GPT-4.1.

| Level | Name | Description | Example |
|:---:|---|---|---|
| 0 | Not actionable | Purely evaluative | *"Good job overall, keep it up!"* |
| 1 | Moderately actionable | Vague suggestion | *"Consider adding more comments to explain your code."* |
| 2 | Highly actionable | Specific instruction | *"On line 45, replace the for-loop with vectorized operations for better performance."* |

Universities often cannot send student work to a third-party API, and per-call LLM pricing does not scale across every comment in every term. A compact encoder fine-tuned on domain-specific data runs locally, costs nothing per call, and — as the results below show — performs comparably to a fine-tuned frontier LLM.

---

## Repository structure

```
Feedback_Actionability/
├── Comparisons/              # 5 training + evaluation notebook（different random seeds） per model
│   ├── albert/
│   ├── bert/
│   ├── deberta/
│   ├── edubert/
│   └── roberta/
├── Synth_Data/
│   └── synthCS.csv           # Synthetic feedback used for class balancing (training only)
├── .gitignore
├── LICENSE
└── README.md
```

## Data

### Included

`Synth_Data/synthCS.csv` contains synthetic feedback comments in computer science domain, generated with GPT-4o to address class imbalance, since highly actionable feedback is rare in authentic data. These were generated using a structured three-part prompt (task definition, rubric with computer-science examples, JSON output constraint), with comments constrained to 10–40 words and stylistic variation enforced to prevent duplication. A random subset of 100 samples was manually reviewed for tonal consistency.

**Synthetic data was used exclusively for training.** The validation and test sets consist entirely of authentic feedback, so no reported metric is measured on generated text.

### Not included

The authentic tutor feedback is **not** redistributed here, as it belongs to its original publishers. It can be obtained directly from:

1. **Menagerie: A Dataset of Graded CS1 Assignments** — Messer, Brown, Kölling & Shi (2025), *ACM Transactions on Computing Education*, 25(4). Graded programming assignments with separate feedback across four criteria: correctness, code elegance, readability, and documentation.
2. **Students' Assignment Feedbacks** — Datadynamo.ai (2024), Kaggle. <https://www.kaggle.com/datasets/datadynamoai/students-assignment-feedbacks> Teacher feedback paired with student responses to short-answer questions.

Labels for the authentic data were produced with the OpenAI batch API using GPT-4o at temperature 0.1 under a structured rubric. A random sample of 250 of 1,887 instances was manually verified by the first author, with 90.0% agreement between the automated and manual labels.

### Dataset composition

| Split | Authentic | Synthetic | Total |
|---|---|---|---|
| Train | 943 | 1,187 | 2,130 |
| Validation | 755 | 0 | 755 |
| Test | 189 | 0 | 189 |

---

## Reproducing the results

### Requirements

```
torch
transformers
scikit-learn
pandas
numpy
```

Training and evaluation were run on a single NVIDIA A100 GPU, taking approximately six minutes per model.

### Training configuration

All encoder models were trained with identical settings:

| Hyperparameter | Value |
|---|---|
| Epochs | 12 |
| Batch size | 12 |
| Gradient accumulation steps | 2 |
| Learning rate | 2 × 10⁻⁵ |
| Optimiser | AdamW, weight decay 1 × 10⁻³ |
| Schedule | Linear warm-up, then linear decay |
| Loss | Cross-entropy |

The classification head takes the hidden state of the `[CLS]` token from the final transformer layer, passes it through a dense pre-classification layer with ReLU activation and dropout (p = 0.1), then a linear layer over the three actionability classes.

Each model was evaluated over five runs with different random seeds to ensure results were not due to chance.

### Steps

1. Obtain the two authentic feedback datasets from the sources listed above.
2. Open the notebook for the model you want to reproduce in `Comparisons/<model>/`.
3. Point the data paths at your local copies of the authentic data and at `Synth_Data/synthCS.csv`.
4. Run training and evaluation.

---

## Using the trained model

Download `pytorch_roberta_actionability.bin` from the [latest release](https://github.com/wanna666/Feedback_Actionability/releases/tag/v1.0), then load it into the classifier defined in the RoBERTa notebook


## Citation

Pending until the conference proceeding has been published.

---

## License

Code in this repository is released under the MIT License.

The authentic feedback datasets are **not** covered by this licence and remain subject to the terms of their original publishers.