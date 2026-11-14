"""
Project 04 - First Machine Learning Classifier.
InternCircle Data Science & AI Internship - Ritun Kumar
Student ID: IC-2026-E21E85D134E22C70

Run:  python ml_classifier.py
Data: classic Titanic dataset (891 passengers), public CSV linked in the
      InternCircle project guide:
      https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
Task: predict whether a passenger survived (Survived = 0/1) from
      Pclass, Sex, Age, SibSp, Parch, Fare and Embarked.
Models: Decision Tree (main model) and Logistic Regression (comparison).
Outputs: printed metrics, charts in outputs/, metrics in outputs/metrics.txt
         and the trained model in models/titanic_survival_model.pkl
         (the same model file used by Project 05, the Streamlit app).
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "outputs")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def load_features():
    df = pd.read_csv(DATA_URL)
    # Cleaning: median for Age/Fare, mode for Embarked, drop Cabin/Name/Ticket.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    # One-hot encode the two text columns (done before the split so train
    # and test have exactly the same columns).
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    return X, y


def evaluate(name, model, X_test, y_test, lines):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    # MSE here is the mean squared error between the true 0/1 labels and the
    # model's predicted survival probabilities (a probability-error measure).
    mse = mean_squared_error(y_test, y_prob)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)
    print(classification_report(y_test, y_pred))
    print("MSE (true labels vs predicted probabilities):", round(mse, 4))
    lines.append(f"{name}")
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append(f"Confusion matrix:\n{cm}")
    lines.append(classification_report(y_test, y_pred))
    lines.append(f"MSE (labels vs predicted probabilities): {mse:.4f}")
    lines.append("")
    return acc, cm


def main():
    X, y = load_features()
    print("Features:", list(X.columns))
    print("Rows:", len(X))
    # Train/test split: 80% train, 20% test, stratified so both sets keep
    # the same survival share.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print("Train:", len(X_train), " Test:", len(X_test))

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    logreg = Pipeline([("scaler", StandardScaler()),
                       ("model", LogisticRegression(max_iter=1000))])
    logreg.fit(X_train, y_train)

    lines = ["Titanic survival classifier - test-set metrics (20% held out, seed 42)", ""]
    tree_acc, tree_cm = evaluate("Decision Tree (max_depth=5)", tree, X_test, y_test, lines)
    logreg_acc, _ = evaluate("Logistic Regression (comparison)", logreg, X_test, y_test, lines)
    print("Better model on test accuracy:",
          "Decision Tree" if tree_acc >= logreg_acc else "Logistic Regression")

    # Confusion matrix chart (Decision Tree)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(tree_cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted died", "Predicted survived"],
                yticklabels=["Actually died", "Actually survived"])
    ax.set_title("Confusion matrix - Decision Tree (test set)")
    save(fig, "confusion_matrix.png")

    # Feature importances (Decision Tree)
    importance = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Feature importances (Decision Tree) ===")
    print(importance.round(3))
    lines.append("Feature importances (Decision Tree):")
    lines.append(str(importance.round(3)))
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot.barh(ax=ax, color="#1f4e79")
    ax.set_title("Which features mattered most? (Decision Tree)")
    ax.set_xlabel("Importance")
    save(fig, "feature_importances.png")

    model_path = os.path.join(MODEL_DIR, "titanic_survival_model.pkl")
    joblib.dump({"model": tree, "features": list(X.columns)}, model_path)
    print("\nSaved model:", model_path)
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write("\n".join(lines))
    print("Saved metrics: outputs/metrics.txt")


if __name__ == "__main__":
    main()
