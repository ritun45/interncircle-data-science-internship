"""
Project 05 - Streamlit Interactive ML Web App.
InternCircle Data Science & AI Internship - Ritun Kumar
Student ID: IC-2026-E21E85D134E22C70

Run locally:  streamlit run app.py
Deploy free:  Streamlit Community Cloud, from this GitHub repo,
              main branch, main file project-05-streamlit-ml-app/app.py

The app loads the Decision Tree model trained in Project 04
(models/titanic_survival_model.pkl). If that file is not present
(for example on a fresh deploy), it trains the same model itself from
the Titanic CSV, so the app still works.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    os.path.join(BASE, "models", "titanic_survival_model.pkl"),
    os.path.join(BASE, "..", "project-04-ml-classifier", "models", "titanic_survival_model.pkl"),
    os.path.join("project-04-ml-classifier", "models", "titanic_survival_model.pkl"),
]


def train_model():
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
    return model


@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            saved = joblib.load(path)
            return saved["model"], saved["features"], f"Loaded saved model from {path}"
    return train_model(), FEATURES, "No saved model file found - trained the model in the app."


def main():
    st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
    st.title("🚢 Titanic Survival Predictor")
    st.write("InternCircle Data Science & AI Internship — Ritun Kumar. "
             "Enter a passenger's details and the Decision Tree model "
             "from Project 04 predicts their chance of survival.")
    model, features, source = load_model()
    st.caption(source)

    st.sidebar.header("Passenger details")
    pclass = st.sidebar.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.sidebar.radio("Sex", ["female", "male"])
    age = st.sidebar.slider("Age (years)", 0, 80, 28)
    sibsp = st.sidebar.slider("Siblings / spouses aboard", 0, 8, 0)
    parch = st.sidebar.slider("Parents / children aboard", 0, 6, 0)
    fare = st.sidebar.slider("Fare (GBP)", 0.0, 512.0, 32.0)
    embarked = st.sidebar.selectbox("Embarked at", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
    port = {"Southampton (S)": "S", "Cherbourg (C)": "C", "Queenstown (Q)": "Q"}[embarked]

    # Feature vector in exactly the order the model was trained on.
    row = pd.DataFrame([{
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Sex_male": 1 if sex == "male" else 0,
        "Embarked_Q": 1 if port == "Q" else 0,
        "Embarked_S": 1 if port == "S" else 0,
    }])[features]

    if st.button("Predict survival"):
        prob = float(model.predict_proba(row)[0][1])
        pred = int(model.predict(row)[0])
        st.subheader(f"Survival probability: {prob:.1%}")
        st.progress(prob)
        if pred == 1:
            st.success("Prediction: this passenger would likely SURVIVE.")
        else:
            st.warning("Prediction: this passenger would likely NOT survive.")
        st.write("Try changing the class or sex: in the Titanic data, "
                 "women and 1st-class passengers survived far more often.")

    if hasattr(model, "feature_importances_"):
        st.subheader("What the model relies on")
        importance = pd.Series(model.feature_importances_, index=features).sort_values()
        st.bar_chart(importance)

    st.caption("Model: Decision Tree (max_depth=5), trained on the classic Titanic dataset (891 passengers). "
               "Source code: https://github.com/ritun45/interncircle-data-science-internship")


if __name__ == "__main__":
    main()
