# Kaggle Competition: Human or AI? - Team 18 

**Team Members:** Pallavi Khanna (pk633), Minghan Gao (mg2328), Peter Chang (pc783) 

## Project Overview 

This project was about developing a binary test classifier by experimenting with different machine learning models to identify whether a passage was AI-generated or written by a human, since it's starting to be difficult to differentiate between an LLM generated passage versus a human generated passage.

## Models 

**For all of the models we experimented, we decided to use a random seed of 42 and split the data into 8,000 training and 2,000 validation examples.**

The first models we experimented with were logistic regression, multimodal Naive Bayes, and SVM which used TF-IDF features for the text. Of these models, Logistic Regression score the highest validation AUROC score of 0.9246 after hyperparameter tuning, however when we made the first submission we received a score of 0.81356 from Kaggle which was lower than our validation score. 

*INSERT OTHER MODELS HERE* 

## How to Run 
