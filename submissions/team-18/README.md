# Kaggle Competition: Human or AI? - Team 18 

**Team Members:** Pallavi Khanna (pk633), Minghan Gao (mg2328), Peter Chang (pc783) 

## Project Overview 

This project was about developing a binary test classifier by experimenting with different machine learning models to identify whether a passage was AI-generated or written by a human, since it's starting to be difficult to differentiate between an LLM generated passage versus a human generated passage.

## Models 

**For all of the models we experimented, we decided to use a random seed of 42 and split the data into 8,000 training and 2,000 validation examples.**

The first models we experimented with were logistic regression, multimodal Naive Bayes, and SVM which used TF-IDF features for the text. Of these models, Logistic Regression score the highest validation AUROC score of 0.9246 after hyperparameter tuning, however when we made the first submission we received a score of 0.81356 from Kaggle which was lower than our validation score. 

### Minghan’s CPU experiments

Minghan compared a decision tree, random forest, histogram gradient boosting, and a multilayer perceptron (MLP) against a separate logistic regression baseline using word and character TF-IDF. The four models used 128 SVD components and 10 writing-style features. The MLP also scaled its inputs. All feature transformations were fitted on the training partition.

| Model | Main settings | Train AUROC | Validation AUROC | Validation 95% CI | Fit and save |
| --- | --- | ---: | ---: | --- | ---: |
| Logistic regression (baseline) | Sparse word and character TF-IDF, max 500 iterations | 0.999239 | 0.964987 | 0.957674–0.972119 | 15.4 s |
| Decision tree | Maximum depth 12, minimum 10 samples per leaf | 0.989933 | 0.963850 | 0.954426–0.972596 | 21.5 s |
| Random forest | 350 trees, minimum 3 samples per leaf | 0.999997 | 0.992921 | 0.990047–0.995223 | 23.5 s |
| Histogram gradient boosting | 250 iterations, learning rate 0.07 | 1.000000 | **0.998082** | 0.996775–0.999039 | 21.0 s |
| MLP | Hidden layers of 256 and 64 units, alpha 0.01, max 45 iterations | 1.000000 | 0.983124 | 0.978277–0.987254 | 20.4 s |

Histogram gradient boosting had the highest validation AUROC in this comparison (**0.998082**). The boosting submission scored approximately **0.87 public AUROC** on Kaggle.
*INSERT OTHER MODELS HERE* 

## How to Run 

As we primarily used the Jupyter Notebook to run our experiments please reference these instructions to run the code.

1. Access and open our experiments (`Experiment_git.ipynb,` `logisticreg_experiment.ipynb,` **`minghan's experiment`**)
2. Install the required libraries:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
3. Upload the test.csv, train.csv, and unlabeled.csv files
4. Run all cells in order so the notebook trains and evaluates all the models.

Python version:
- For `logisticreg_experiment.ipynb,` - Python version 3.13.18

### Minghan’s CPU experiments

Run these commands from `submissions/team-18/`. The experiments were run with Python 3.13.5 on a laptop CPU.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `data/` folder inside `submissions/team-18/` and place the competition’s `train.csv` there. Keep the CSV in its original row order to reproduce the same stratified split of 8,000 training and 2,000 validation examples.

Run the logistic regression baseline and the four model comparisons:

```bash
python train.py --train data/train.csv --model linear --seed 42 --run-dir models/val-linear
python train.py --train data/train.csv --model tree --seed 42 --run-dir models/val-tree
python train.py --train data/train.csv --model forest --seed 42 --run-dir models/val-forest
python train.py --train data/train.csv --model boost --seed 42 --run-dir models/val-boost
python train.py --train data/train.csv --model mlp --seed 42 --run-dir models/val-mlp
```

Each run saves its configuration, AUROC, and runtime in `run.json` inside its output directory. It also saves the validation IDs and scores in `validation_predictions.csv`. Use a fresh `--run-dir` when rerunning an experiment.

### Instructions for Final Model/Submission.csv

## AI Disclosure

For this assignment, we used AI tools like ChatGPT to help us understand concepts, as a coding aid, and to help
with LaTeX formatting. It helped explain equations and what some questions were asking, how to
approach problems, and developing and troubleshooting Python code.

