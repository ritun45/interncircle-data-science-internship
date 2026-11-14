# Project 02 — Data Visualization with Matplotlib & Seaborn

**InternCircle Data Science & AI Internship — Ritun Kumar (IC-2026-E21E85D134E22C70)**

## Aim

Turn the Titanic dataset into clear, well-labelled charts that tell the story of who survived and why, using Matplotlib and Seaborn with a custom colour palette.

## Files

- `visualization.py` — creates all six charts.
- `outputs/` — created when the script runs: the six PNG charts.

## How to run

```bash
pip install pandas numpy matplotlib seaborn
python visualization.py
```

Data: classic Titanic dataset (891 passengers), public CSV linked in the InternCircle project guide: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

## Charts produced

1. `01_age_histogram.png` — histogram of age, split by survival, with KDE curve.
2. `02_survival_by_class_bar.png` — bar chart of survival rate by passenger class, with percentages labelled.
3. `03_age_vs_fare_scatter.png` — scatter plot of age vs fare, coloured by survival.
4. `04_fare_by_class_boxplot.png` — box plot of fare by passenger class.
5. `05_correlation_heatmap.png` — correlation heatmap of the numeric columns.
6. `06_embarked_by_sex_bar.png` — grouped bar chart of passengers by embarkation port and sex.

## Design choices

- Custom palette (deep blue, teal, warm orange) used consistently; survival is always teal, death is red-brown, so charts can be read together.
- Every chart has a title, labelled axes and a readable scale; the bar chart labels each value as a percentage.

## What the charts show (storytelling)

- Survival was not random: women and 1st-class passengers survived far more often.
- Most passengers were young adults (20-40); children survived better than adults.
- Fares were very unequal: 1st-class fares were far higher and more spread out than 3rd-class fares.
- The heatmap shows survival rising with being female and with higher class.
- Most passengers boarded at Southampton.

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn.
