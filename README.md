# Telco Customer Churn Prediction

Machine Learning project for predicting customer churn in a telecommunications company using a complete classification workflow, from exploratory data analysis and preprocessing to model comparison and deployment recommendation.

## Project Overview

The project uses the Telco Customer Churn dataset, containing 7,043 customers and 21 original columns.

The objective is to predict whether a customer will churn while considering not only predictive performance, but also class imbalance, false-positive and false-negative errors, business cost, model interpretability, and generalization.

## Machine Learning Workflow

The project includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Leakage-safe pipelines
- Numerical scaling and categorical encoding
- Feature selection using Mutual Information and L1 regularization
- Dummy baseline evaluation
- Stratified 5-fold cross-validation
- Hyperparameter tuning
- Class imbalance handling using class weighting and SMOTE
- ROC curve and confusion matrix analysis
- Cost-sensitive threshold optimization
- Data leakage experiment
- Final model comparison
- Deployment recommendation

## Models Evaluated

Five classification algorithms were investigated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- LightGBM

## Final Model Performance

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.8062 | 0.6747 | 0.5214 | 0.5882 | 0.8473 |
| XGBoost | 0.8048 | 0.6713 | 0.5187 | 0.5852 | 0.8450 |
| Logistic Regression | 0.8048 | 0.6552 | 0.5588 | 0.6032 | 0.8411 |
| Random Forest | 0.7921 | 0.6753 | 0.4171 | 0.5157 | 0.8398 |
| SVM | 0.7921 | 0.6484 | 0.4733 | 0.5471 | 0.8299 |

LightGBM achieved the highest test ROC-AUC, while XGBoost achieved the highest cross-validation ROC-AUC.

## Business-Cost Analysis

The assignment assumes:

- False Negative cost = $500
- False Positive cost = $50

Because missing a real churner is substantially more expensive than unnecessarily targeting a non-churner, decision thresholds were optimized using out-of-fold training predictions.

| Model | Threshold | FP | FN | Recall | Total Cost |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.08 | 566 | 12 | 0.9679 | $34,300 |
| LightGBM | 0.07 | 618 | 9 | 0.9759 | $35,400 |
| XGBoost | 0.10 | 537 | 19 | 0.9492 | $36,350 |

## Deployment Recommendation

The recommended deployment model is **cost-sensitive Logistic Regression with a decision threshold of 0.08**.

Although LightGBM achieved the highest test ROC-AUC, Logistic Regression produced the lowest measured business cost while maintaining very high churn recall.

It also provides important deployment advantages:

- High interpretability
- Low computational complexity
- Fast training and prediction
- Strong generalization
- Low business cost
- Simple implementation and maintenance

The final model therefore balances predictive performance with operational and business requirements rather than selecting a model based only on accuracy.

## Leakage Prevention

All learned preprocessing operations were incorporated into machine-learning pipelines during cross-validation.

An intentional leakage experiment was also performed to demonstrate why feature selection and other learned transformations must be fitted independently inside each training fold.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- imbalanced-learn
- Matplotlib
- Jupyter Notebook

## Reproducibility

A fixed random seed (`random_state=42`) was used throughout applicable data splits, models, resampling procedures, and tuning experiments.

## Files

- `ML_Assignment_1.ipynb` - complete machine-learning implementation and experiments
- `ML_Assignment_Donia_Final.pdf` - final project report
- `requirements.txt` - Python dependencies
- `README.md` - project documentation

## Author

**Donia Said**  
Machine Learning Assignment 1  
Birzeit University
