# Project 04 — First Machine Learning Classifier

**InternCircle Data Science & AI Internship — Ritun Kumar (IC-2026-E21E85D134E22C70)**

## Aim

Build and evaluate a first machine-learning model with Scikit-Learn that predicts whether a Titanic passenger survived, and report its accuracy, confusion matrix and MSE.

## Files

- `ml_classifier.py` — loads and cleans the data, trains and evaluates the models, saves the model and metrics.
- `outputs/` — created when the script runs: `metrics.txt`, `confusion_matrix.png`, `feature_importances.png`.
- `models/` — created when the script runs: `titanic_survival_model.pkl` (Decision Tree model + feature list, also used by Project 05).

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python ml_classifier.py
```

Data: classic Titanic dataset (891 passengers), public CSV linked in the InternCircle project guide: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

## Steps in the script

1. Clean the data: fill missing Age and Fare with their medians, fill missing Embarked with the most common port, and drop PassengerId, Name, Ticket and Cabin.
2. One-hot encode Sex and Embarked (before the split, so train and test have the same columns).
3. Split into train/test: 80% train, 20% test, stratified (random_state 42) so both sets keep the same survival share.
4. Train a Decision Tree Classifier (max_depth 5) as the main model, and a Logistic Regression (with scaling) as a comparison.
5. Evaluate both on the test set: Accuracy, Confusion Matrix, classification report, and MSE between the true 0/1 labels and the predicted survival probabilities.
6. Chart the confusion matrix and the Decision Tree feature importances, and save the trained model with joblib.

## Results (verified by running the script; test set = 179 passengers)

- **Decision Tree (max_depth=5): Accuracy 0.7654, MSE (true labels vs predicted probabilities) 0.1678.** Confusion matrix (rows = actual died/survived, columns = predicted died/survived): [[97, 13], [29, 40]].
- **Logistic Regression (comparison): Accuracy 0.8045, MSE 0.1450.** Confusion matrix: [[98, 12], [23, 46]].
- **Feature importances (Decision Tree):** Sex_male 0.531, Pclass 0.180, Age 0.129, Fare 0.114, Embarked_S 0.034, SibSp 0.011, Parch 0.001, Embarked_Q 0.000.
- So sex was by far the strongest predictor, followed by passenger class, age and fare — the same pattern found in the Project 01 EDA. The exact numbers are also written to `outputs/metrics.txt` when the script runs.

## Tools used

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, joblib.
