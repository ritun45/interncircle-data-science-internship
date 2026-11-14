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

## Results

The exact test-set numbers are printed by the script and written to `outputs/metrics.txt` — run the script to reproduce them. The Decision Tree and Logistic Regression accuracies are compared in the script output, and the feature-importance chart shows which inputs (sex, class, fare, age) the tree relied on most.

## Tools used

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, joblib.
