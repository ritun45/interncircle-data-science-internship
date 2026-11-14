"""
Project 05 - Train and save the model used by the Streamlit app.
InternCircle Data Science & AI Internship - Ritun Kumar
Student ID: IC-2026-E21E85D134E22C70

Run:  python train_model.py
Then: streamlit run app.py

Trains the same Decision Tree model as Project 04 (max_depth=5) on the
Titanic dataset and saves it with joblib to models/titanic_survival_model.pkl
in this folder, together with the feature list, so app.py can load it.
"""

import os

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]


def main():
    df = pd.read_csv(DATA_URL)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    X = df[FEATURES]
    y = df["Survived"]
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    print("Trained on", len(X), "passengers. Training accuracy:", round(model.score(X, y), 4))
    path = os.path.join(MODEL_DIR, "titanic_survival_model.pkl")
    joblib.dump({"model": model, "features": FEATURES}, path)
    print("Saved model:", path)


if __name__ == "__main__":
    main()
