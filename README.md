# Cross-Lingual Intent Classification: English, Urdu & Bengali

**A reproducible multilingual NLP benchmarking project using PyTorch, Hugging Face Transformers, and Amazon MASSIVE.**

This project investigates a practical question: **How reliably do multilingual language models transfer intent-classification ability from English to Urdu and Bengali without target-language fine-tuning?** It compares **mBERT** (`bert-base-multilingual-cased`) and **XLM-RoBERTa** (`xlm-roberta-base`) using English-only supervised fine-tuning and zero-shot evaluation on two target languages. The analysis covers performance, intent-level errors, tokenization, random-seed variability, and paired statistical comparisons.

> **Project type:** NLP / multilingual transfer learning / empirical model evaluation. This is a portfolio engineering and analysis project, not a claim of a published research paper.

## Highlights

- **English-only training; zero-shot Urdu and Bengali testing.** No Urdu or Bengali labeled examples were used for fine-tuning.
- **60-class intent classification** on Amazon MASSIVE v1.0, with **2,974 held-out test utterances per language**.
- **Three training seeds** (21, 42, 123) per model, with accuracy and macro-F1 reported as mean ± sample standard deviation.
- **XLM-RoBERTa achieved 58.15% Urdu and 58.41% Bengali mean accuracy**, compared with 26.06% and 25.81% for mBERT.
- Paired McNemar tests, Holm correction, and paired bootstrap confidence intervals support XLM-RoBERTa's advantage on the evaluated Urdu and Bengali test examples.

## Results

### Test accuracy

| Language | mBERT (mean ± SD) | XLM-RoBERTa (mean ± SD) | XLM-R − mBERT |
|---|---:|---:|---:|
| English | 0.8366 ± 0.0012 | **0.8448 ± 0.0036** | +0.0082 |
| Urdu | 0.2606 ± 0.0148 | **0.5815 ± 0.0151** | +0.3209 |
| Bengali | 0.2581 ± 0.0031 | **0.5841 ± 0.0041** | +0.3259 |

### Macro-F1 (all 60 classes)

| Language | mBERT (mean ± SD) | XLM-RoBERTa (mean ± SD) | XLM-R − mBERT |
|---|---:|---:|---:|
| English | **0.7067 ± 0.0022** | 0.6922 ± 0.0173 | −0.0146 |
| Urdu | 0.1653 ± 0.0174 | **0.4219 ± 0.0036** | +0.2565 |
| Bengali | 0.1701 ± 0.0157 | **0.4314 ± 0.0181** | +0.2613 |

**Metric convention:** Macro-F1 is calculated with an explicit list of all **60 intent labels** and `zero_division=0`. Only **59 intents occur in the held-out test labels**; `cooking_query` has no ground-truth test examples. The full 60-class convention is used consistently in the verified results.

**Interpretation:** English accuracy is similar across models, but XLM-RoBERTa transfers substantially better to Urdu and Bengali. English macro-F1 does not show a consistent XLM-RoBERTa advantage.

## Dataset

[Amazon MASSIVE](https://github.com/alexa/massive) (v1.0) is a multilingual, task-oriented natural-language understanding dataset.

| Language | Locale | Train | Validation | Test |
|---|---|---:|---:|---:|
| English | `en-US` | 11,514 | 2,033 | 2,974 |
| Urdu | `ur-PK` | 11,514 | 2,033 | 2,974 |
| Bengali | `bn-BD` | 11,514 | 2,033 | 2,974 |

Only **English training and validation** partitions were used for model fitting and checkpoint selection. All three languages' **test** partitions were used for final evaluation. The target-language training partitions were not used for fine-tuning.

Please obtain the dataset from the original provider and follow its license and terms; the full dataset is not redistributed in this repository.

## Experimental design

1. **Fine-tune mBERT and XLM-RoBERTa** for 60-way intent classification using English training examples.
2. **Select checkpoints** using English validation macro-F1; do not select checkpoints using Urdu/Bengali test performance.
3. **Evaluate zero-shot transfer** on English, Urdu, and Bengali held-out test partitions.
4. **Analyze errors** using per-intent metrics, incorrect predictions, and paired model comparisons.
5. **Analyze tokenization** using sequence length and subwords-per-word statistics.
6. **Repeat with three random seeds** (21, 42, 123) and report mean ± sample SD.
7. **Verify checkpoint integrity**, repair legacy mBERT LayerNorm parameter names where necessary, and recompute metrics from saved models.
8. **Compare paired predictions** with exact McNemar tests, Holm correction, and 2,000 paired bootstrap resamples per seed/language.

### Model/training settings

| Setting | Value |
|---|---|
| mBERT backbone | `bert-base-multilingual-cased` |
| XLM-R backbone | `xlm-roberta-base` |
| Training language | English only |
| Epochs | 3 |
| Learning rate | 2 × 10⁻⁵ |
| Train batch size | 16 |
| Evaluation batch size | 32 (multi-seed runs) |
| Warmup steps | 216 |
| Weight decay | 0.01 |
| Maximum input length | 64 tokens (verified evaluation) |
| Random seeds | 21, 42, 123 |
| Hardware for multi-seed runs | NVIDIA Tesla T4 (Google Colab) |
| Selection metric | English validation macro-F1 |

**Reproducibility note:** The original seed-42 checkpoints were produced in earlier runs, while seeds 21 and 123 were produced in the later multi-seed workflow. The original mBERT run used CPU/full-precision training, whereas the later GPU runs used mixed precision. Treat three-seed summaries as descriptive rather than a perfectly controlled estimate of seed-only variability. Before asserting bit-for-bit reproduction, verify the original notebooks, library versions, and preprocessing settings.

## Statistical analysis

For each language and seed, both models were evaluated on the **same 2,974 examples**, aligned by example ID.

- **Exact McNemar test:** paired correctness/accuracy comparison for each fixed model pair.
- **Holm correction:** applied across nine McNemar comparisons (3 languages × 3 seeds).
- **Paired bootstrap:** 2,000 resamples per language/seed, estimating 95% percentile intervals for accuracy and macro-F1 differences.
- **Difference convention:** XLM-RoBERTa minus mBERT; positive values favor XLM-RoBERTa.

| Language | Holm-significant accuracy comparisons | Macro-F1 difference positive in |
|---|---:|---:|
| English | 0 / 3 | 1 / 3 seeds |
| Urdu | **3 / 3** | **3 / 3 seeds** |
| Bengali | **3 / 3** | **3 / 3 seeds** |

All six Urdu/Bengali per-seed paired bootstrap **95% macro-F1 intervals exclude zero in the positive direction**. These intervals are *not* multiple-comparison-adjusted and represent uncertainty from resampling test examples for fixed checkpoints—not uncertainty over all possible training seeds. The tests do not establish that tokenization differences *cause* performance differences.

## Tokenization findings

In the seed-42 analysis, XLM-RoBERTa generated shorter sequences than mBERT, particularly for the two target languages:

| Language | mBERT mean tokens | XLM-R mean tokens | Approx. reduction |
|---|---:|---:|---:|
| English | 10.459 | 10.030 | 4.1% |
| Urdu | 15.845 | 11.900 | 24.9% |
| Bengali | 17.795 | 12.318 | 30.8% |

This is an **observational association**, not evidence that tokenization alone explains the accuracy gap. Sequence length, pretraining data, representations, and other architectural/training differences may all contribute.

## Repository layout

The recommended layout is shown below. **Add the files only after exporting and checking the actual Colab notebooks and result files**; the layout is not a claim that every file has already been uploaded.

```text
cross-lingual-intent-evaluation/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   ├── 01_mbert_training.ipynb
│   ├── 02_xlmr_training.ipynb
│   ├── 03_error_analysis.ipynb
│   ├── 04_tokenization_analysis.ipynb
│   ├── 05_multi_seed_experiment.ipynb
│   ├── 06_checkpoint_verification.ipynb
│   └── 07_statistical_analysis.ipynb
├── results/
│   ├── final_mean_std.csv
│   ├── three_seed_statistical_summary.csv
│   └── publication_ready_summary.csv
└── figures/
    ├── model_comparison.png
    └── macro_f1_bootstrap_ci.png
```

## Reproducing the experiments

The primary development environment was **Google Colab**. Once the notebooks are uploaded:

1. Create a Python environment and install the dependencies listed in `requirements.txt` (or run the installation cell in the notebooks).
2. Download MASSIVE v1.0 from the [official repository](https://github.com/alexa/massive) and follow its data access instructions.
3. Open the notebooks in numerical order, beginning with English-only training.
4. Replace any author-specific `/content/drive/MyDrive/...` paths with paths appropriate for your environment.
5. Use the same intent-label mapping, splits, tokenization limits, random seeds, and **60-label macro-F1** convention.
6. Run the checkpoint verification and statistical analysis notebooks after generating predictions.

**Current portability limitation:** The original notebooks were developed with Google Drive paths and may need path/configuration cleanup before they can be executed from a fresh checkout. Large model checkpoints are not included by default. The final result tables above come from the completed Colab runs.

## Technology stack

**Python · PyTorch · Hugging Face Transformers · Hugging Face Datasets · scikit-learn · pandas · NumPy · statsmodels · Matplotlib · Google Colab**

## Limitations

- Only **two pretrained backbones** and **three languages** were evaluated.
- The evaluation is **English-to-target zero-shot transfer**, not multilingual fine-tuning or deployment in a production assistant.
- Three seeds provide useful descriptive robustness checks but limited evidence about training-run variability.
- Original seed-42 training conditions were not fully identical to the later multi-seed GPU runs.
- `cooking_query` has no ground-truth examples in the held-out test partition; the reported macro-F1 nevertheless uses all 60 configured labels.
- McNemar significance is about **paired accuracy** for fixed checkpoints; bootstrap macro-F1 intervals were not multiplicity-adjusted.
- Tokenization statistics do not establish a causal mechanism for cross-lingual performance differences.

## Acknowledgments and attribution

- Dataset: [Amazon MASSIVE](https://github.com/alexa/massive).
- Pretrained backbones: [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased) and [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base).
- Libraries: [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://github.com/huggingface/transformers).

## Author

**Fatima Noor** — Computer Science graduate and AI/NLP research practitioner.

GitHub: [Fatimanoor123](https://github.com/Fatimanoor123)

---

*This repository documents an independent empirical portfolio project. Results should be interpreted with the experimental limitations stated above.*
