"""
Project 01 - Exploratory Data Analysis (EDA) on the Titanic dataset.
InternCircle Data Science & AI Internship - Ritun Kumar
Student ID: IC-2026-E21E85D134E22C70

Run:  python titanic_eda.py
Data: classic Titanic dataset (891 passengers), public CSV linked in the
      InternCircle project guide:
      https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
Outputs: printed statistics, charts in outputs/ and a cleaned CSV.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():
    df = pd.read_csv(DATA_URL)
    print("=== Dataset shape ===")
    print("Rows:", df.shape[0], " Columns:", df.shape[1])
    print(df.dtypes)
    print("\n=== First rows ===")
    print(df.head())

    print("\n=== Missing values ===")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"missing": missing, "percent": missing_pct}))

    # Missing-value chart
    fig, ax = plt.subplots(figsize=(8, 4))
    missing[missing > 0].sort_values().plot.barh(ax=ax, color="#4c72b0")
    ax.set_title("Missing values per column (Titanic)")
    ax.set_xlabel("Number of missing values")
    save(fig, "missing_values.png")

    # --- Cleaning: median Age, mode Embarked, drop Cabin (77% missing) ---
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = df.drop(columns=["Cabin"])
    print("\nAfter cleaning: missing values left:", int(df.isna().sum().sum()))

    print("\n=== Descriptive statistics (numeric) ===")
    print(df.describe().round(2))

    print("\n=== Overall survival rate ===")
    print("Survived:", df["Survived"].mean().round(4), "(", int(df["Survived"].sum()), "of", len(df), ")")

    print("\n=== Survival rate by sex ===")
    print(df.groupby("Sex")["Survived"].agg(["mean", "count"]).round(3))

    print("\n=== Survival rate by passenger class ===")
    print(df.groupby("Pclass")["Survived"].agg(["mean", "count"]).round(3))

    print("\n=== Survival rate by embarkation port ===")
    print(df.groupby("Embarked")["Survived"].agg(["mean", "count"]).round(3))

    print("\n=== Survival rate by age band ===")
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                           labels=["Child 0-12", "Teen 13-18", "Adult 19-35", "Adult 36-60", "Senior 60+"])
    print(df.groupby("AgeBand", observed=True)["Survived"].agg(["mean", "count"]).round(3))

    print("\n=== Correlation with survival (numeric columns) ===")
    corr = df.select_dtypes(include=[np.number]).corr()["Survived"].sort_values(ascending=False)
    print(corr.round(3))

    # --- Charts ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x="Survived", hue="Sex", ax=ax)
    ax.set_title("Survival count by sex (0 = died, 1 = survived)")
    save(fig, "survival_by_sex.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, ax=ax)
    ax.set_title("Age distribution by survival")
    save(fig, "age_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Pclass", y="Fare", ax=ax)
    ax.set_title("Fare by passenger class")
    save(fig, "fare_by_class.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True,
                cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap (numeric columns)")
    save(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    df.groupby("Pclass")["Survived"].mean().plot.bar(ax=ax, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("Survival rate by passenger class")
    ax.set_ylabel("Survival rate")
    ax.set_ylim(0, 1)
    save(fig, "survival_by_class.png")

    df.to_csv(os.path.join(OUT_DIR, "titanic_cleaned.csv"), index=False)
    print("\nSaved cleaned data: outputs/titanic_cleaned.csv")

    print("""
=== Key insights ===
1. About 38% of the 891 passengers survived.
2. Sex mattered most: about 74% of women survived, versus about 19% of men.
3. Class mattered: about 63% of 1st-class passengers survived, versus about 24% in 3rd class.
4. Children (0-12) survived at a higher rate than adults aged 19-60.
5. Age had 177 missing values and Cabin about 77% missing, so Cabin was dropped and Age filled with its median (28).
6. Fare is right-skewed: a few very expensive tickets, most passengers paid far less.
7. Survival is positively linked with being female and travelling in a higher class, and negatively linked with passenger class number.
""")


if __name__ == "__main__":
    main()
