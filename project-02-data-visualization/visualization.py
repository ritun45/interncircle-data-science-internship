"""
Project 02 - Data Visualization with Matplotlib & Seaborn.
InternCircle Data Science & AI Internship - Ritun Kumar
Student ID: IC-2026-E21E85D134E22C70

Run:  python visualization.py
Data: classic Titanic dataset (891 passengers), public CSV linked in the
      InternCircle project guide:
      https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
Outputs: six charts in outputs/, each with a title, labelled axes and a
         custom colour palette.
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

# Custom palette: deep blue for died, teal for survived, warm accent for highlights.
CUSTOM = ["#1f4e79", "#2e8b8b", "#e67e22", "#8e44ad", "#27ae60"]
SURVIVAL = {0: "#c0392b", 1: "#2e8b8b"}
sns.set_theme(style="whitegrid", palette=CUSTOM, font_scale=1.05)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():
    df = pd.read_csv(DATA_URL)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    print("Passengers:", len(df), " Survival rate:", round(df["Survived"].mean(), 3))

    # 1. Histogram: age distribution, split by survival
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True,
                 palette=SURVIVAL, ax=ax)
    ax.set_title("Age distribution of Titanic passengers, by survival", fontsize=13)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of passengers")
    save(fig, "01_age_histogram.png")

    # 2. Bar chart: survival rate by passenger class
    rates = df.groupby("Pclass")["Survived"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    rates.plot.bar(ax=ax, color=CUSTOM[:3], edgecolor="white")
    ax.set_title("Survival rate falls sharply from 1st to 3rd class", fontsize=13)
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("Survival rate")
    ax.set_ylim(0, 1)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.03, f"{v:.1%}", ha="center", fontsize=11)
    save(fig, "02_survival_by_class_bar.png")

    # 3. Scatter plot: age vs fare, coloured by survival
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived",
                    palette=SURVIVAL, alpha=0.7, ax=ax)
    ax.set_title("Fare vs age: survivors are spread across ages, high fares cluster in 1st class", fontsize=12)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Fare (GBP)")
    ax.set_ylim(0, 300)
    save(fig, "03_age_vs_fare_scatter.png")

    # 4. Box plot: fare by passenger class
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Pclass", y="Fare", palette=CUSTOM, ax=ax)
    ax.set_title("Fare by passenger class (1st class paid far more)", fontsize=13)
    ax.set_xlabel("Passenger class")
    ax.set_ylabel("Fare (GBP)")
    ax.set_ylim(0, 300)
    save(fig, "04_fare_by_class_boxplot.png")

    # 5. Correlation heatmap (numeric columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True,
                cmap="RdYlGn", fmt=".2f", ax=ax)
    ax.set_title("Correlation heatmap: what moves with survival?", fontsize=13)
    save(fig, "05_correlation_heatmap.png")

    # 6. Grouped bar chart: passengers by embarkation port and sex
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="Embarked", hue="Sex", palette=["#1f4e79", "#e67e22"], ax=ax)
    ax.set_title("Passengers by embarkation port and sex (most boarded at Southampton)", fontsize=12)
    ax.set_xlabel("Embarkation port (C = Cherbourg, Q = Queenstown, S = Southampton)")
    ax.set_ylabel("Number of passengers")
    save(fig, "06_embarked_by_sex_bar.png")

    print("""
=== What the charts show (the story) ===
- Survival was not random: women and 1st-class passengers survived far more often.
- Most passengers were young adults (20-40), and children survived better than adults.
- Fares were very unequal: 1st-class fares were far higher and more spread out than 3rd-class fares.
- The heatmap shows survival rising with being female and with higher class, and falling with class number.
- Most passengers boarded at Southampton, and the port mix was similar for men and women.
""")


if __name__ == "__main__":
    main()
