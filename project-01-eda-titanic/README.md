# Project 01 — Exploratory Data Analysis (EDA) on the Titanic Dataset

**InternCircle Data Science & AI Internship — Ritun Kumar (IC-2026-E21E85D134E22C70)**

## Aim

Explore the classic Titanic passenger dataset (891 passengers) with Python and Pandas: clean the data, describe it with statistics, and find what affected a passenger's chance of survival.

## Files

- `titanic_eda.py` — the full analysis script. Run it to reproduce every result and chart.
- `outputs/` — created when the script runs: 5 charts and `titanic_cleaned.csv`.

## How to run

```bash
pip install pandas numpy matplotlib seaborn
python titanic_eda.py
```

The script loads the dataset from the public CSV linked in the InternCircle project guide: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

## Steps in the script

1. Load the data and check its shape, column types and first rows.
2. Count missing values per column and chart them.
3. Clean the data: fill missing Age with the median (28), fill missing Embarked with the most common port, and drop Cabin (about 77% missing).
4. Describe the numeric columns (mean, median, spread, min/max).
5. Compare survival rates by sex, passenger class, embarkation port and age band.
6. Compute correlations with survival and draw a correlation heatmap.
7. Save 5 charts and the cleaned dataset.

## Findings (printed by the script)

- About 38% of the 891 passengers survived (342 of 891).
- Sex mattered most: about 74% of women survived, versus about 19% of men.
- Class mattered: about 63% of 1st-class passengers survived, versus about 24% in 3rd class.
- Children (0-12) survived at a higher rate than adults aged 19-60.
- Age had 177 missing values; Cabin had about 77% missing, so it was dropped rather than guessed.
- Fare is right-skewed: most passengers paid far less than a few very expensive tickets.
- Survival is positively linked with being female and travelling in a higher class.

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn.
