# Team18 | Human or AI?

**Kaggle team name** `Team18`  
**Team members** Minghan Gao (add NetID) and [add teammate names and NetIDs]

Minghan ran the decision tree, random forest, histogram gradient boosting, and multilayer perceptron (MLP) experiments. Logistic regression served as the reference baseline. All five models were trained and evaluated on a laptop CPU.

## Repository contents

- [Minghan’s CPU experiments](#train-and-validate) use the Python scripts below.
- [Experiment_git.ipynb](Experiment_git.ipynb) contains teammate experiments with text representations, formatting features, and unlabeled data. [logisticreg_experiment.ipynb](logisticreg_experiment.ipynb) contains logistic regression, Naive Bayes, and linear SVM comparisons. The notebooks read CSVs from their working directory.

## Setup

Run these commands from the team folder under `submissions/`. The recorded runs used Python **3.13.5** and the versions pinned in `requirements.txt`, including SciPy **1.17.1** and scikit-learn **1.6.1**.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download `train.csv`, `test.csv`, and `sample_submission.csv` from Kaggle and place them in `data/` inside the team folder. Keep the original row order in `train.csv`. The label is `0` for human and `1` for AI. Keep the input data, fitted models, and generated submission CSV local. [Competition data](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data)

## Train and validate

```bash
python train.py --train data/train.csv --model linear --seed 42 --run-dir models/val-linear
python train.py --train data/train.csv --model tree   --seed 42 --run-dir models/val-tree
python train.py --train data/train.csv --model forest --seed 42 --run-dir models/val-forest
python train.py --train data/train.csv --model boost  --seed 42 --run-dir models/val-boost
python train.py --train data/train.csv --model mlp    --seed 42 --run-dir models/val-mlp
```

Use a fresh `--run-dir` for each rerun. Each validation run saves metrics and settings to `run.json`, split membership to `split.csv`, and per-example validation scores to `validation_predictions.csv`.

| Model | Main settings | Train AUROC | Validation AUROC | Validation 95% CI | Fit and save |
| --- | --- | ---: | ---: | --- | ---: |
| Logistic regression (baseline) | Sparse word and character TF-IDF, max 500 iterations | 0.999239 | 0.964987 | 0.957674–0.972119 | 15.4 s |
| Decision tree | Maximum depth 12, minimum 10 samples per leaf | 0.989933 | 0.963850 | 0.954426–0.972596 | 21.5 s |
| Random forest | 350 trees, minimum 3 samples per leaf | 0.999997 | 0.992921 | 0.990047–0.995223 | 23.5 s |
| Histogram gradient boosting | 250 iterations, learning rate 0.07 | 1.000000 | **0.998082** | 0.996775–0.999039 | 21.0 s |
| MLP | Hidden layers of 256 and 64 units, alpha 0.01, max 45 iterations | 1.000000 | 0.983124 | 0.978277–0.987254 | 20.4 s |

## Predict and write the boosting submission

Histogram gradient boosting led Minghan’s CPU comparison. The submission pipeline refits it on all 10,000 labeled passages and generates one AI probability per test ID in the sample submission's order.

```bash
python train.py --train data/train.csv --model boost --seed 42 --full \
  --run-dir models/final-boost
python predict.py --test data/test.csv --sample-submission data/sample_submission.csv \
  --run-dir models/final-boost --out submission.csv
```

The generated CSV has 3,000 rows with columns `id` and `label`, where `label` contains the predicted probability of AI authorship. The reported Kaggle public AUROC for boosting was approximately **0.87**. Kaggle includes unfamiliar generators and domains, so the local validation AUROC and public leaderboard AUROC reflect different evaluation sets.

## Seeds and evaluation notes

- All comparisons use the same stratified 8,000/2,000 split of `train.csv` with seed **42**. Both teammate notebooks select the same validation IDs on the supplied CSV, including the notebook that sorts by ID before splitting. Each run fitted on 8,000 rows. The duplicate check found zero training passages matching validation text after case and whitespace normalization.
- TF-IDF uses word 1–2 grams and character 3–5 grams. The tree, forest, boosting, and MLP use 128 truncated SVD components from TF-IDF plus 10 writing-style features. The MLP also scales its inputs. The pipeline fits TF-IDF, SVD, scaling, and model parameters on the training rows during validation runs.
- AUROC uses continuous AI probabilities. The intervals come from 1,000 stratified bootstrap resamples of the holdout. Fit and save time includes feature fitting and model serialization.

## AI assistance

Minghan used ChatGPT Codex to help implement the training and prediction scripts, troubleshoot a macOS SciPy import error, verify saved validation predictions, and draft this README. Minghan ran the five CPU comparisons on the competition data and submitted the boosting predictions to Kaggle. Add any teammate disclosures here and use the same account in report Q6.3.
