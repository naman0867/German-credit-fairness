# German Credit Risk Prediction with Fairness Analysis

A machine learning project that predicts whether a loan applicant is a good or bad credit risk using the German Credit dataset, and checks the models for fairness across different groups of applicants.

## Dataset

- Source: [UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
- Size: 1,000 applicants, 20 features, 1 target (`class`)
- Features include: checking status, loan duration, credit history, purpose, credit amount, savings, employment, age, housing, job, and more

## Models Used

- Logistic Regression
- Random Forest
- XGBoost

## Evaluation Metrics

- Accuracy
- ROC-AUC
- F1 Score

## Fairness Analysis

Fairness is assessed with [Fairlearn](https://fairlearn.org/). The goal is to check whether model performance and decisions differ across groups of applicants, so that the model does not unfairly favor or disadvantage any group.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Fairlearn, Matplotlib, Seaborn

## How to Run

1. Download this repository (Code, then Download ZIP) or open the notebook directly in Google Colab.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run all cells in the notebook. The dataset is downloaded automatically from the UCI website.

## Project Structure

```
├── german-credit-notebook.ipynb   # Data analysis, modeling, and fairness evaluation
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview
```
