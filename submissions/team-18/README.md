# Kaggle Competition: Human or AI? - Team 18 

**Team Members:** Pallavi Khanna (pk633), Minghan Gao (mg2328), Peter Chang (pc783) 

## Project Overview 

This project was about developing a binary test classifier by experimenting with different machine learning models to identify whether a passage was AI-generated or written by a human, since it's starting to be difficult to differentiate between an LLM generated passage versus a human generated passage.

## Models 

**For all of the models we experimented, we decided to use a random seed of 42 and split the data into 8,000 training and 2,000 validation examples.**

The first models we experimented with were logistic regression, multimodal Naive Bayes, and SVM which used TF-IDF features for the text. Of these models, Logistic Regression score the highest validation AUROC score of 0.9246 after hyperparameter tuning, however when we made the first submission we received a score of 0.81356 from Kaggle which was lower than our validation score. 

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

### Instructions for Final Model/Submission.csv

## AI Disclosure

For this assignment, we used AI tools like ChatGPT to help us understand concepts, as a coding aid, and to help
with LaTeX formatting. It helped explain equations and what some questions were asking, how to
approach problems, and developing and troubleshooting Python code.

