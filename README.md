# InternCircle — Data Science & AI Internship

Projects by **Ritun Kumar** (B.Sc. Zoology, final year, RLSY College, Ranchi University) for the InternCircle Data Science & AI internship.

- InternCircle Student ID: `IC-2026-E21E85D134E22C70`
- Track: Data Science & AI, 4 weeks, self-paced, remote
- GitHub: https://github.com/ritun45/interncircle-data-science-internship

## Projects (in order)

| # | Project | Folder | What it covers |
|---|---------|--------|----------------|
| 1 | Exploratory Data Analysis (EDA) on Titanic | `project-01-eda-titanic/` | Pandas, statistics, missing values, plots, insights |
| 2 | Data Visualization with Matplotlib & Seaborn | `project-02-data-visualization/` | Bar, scatter, box, heatmap charts, custom palettes |
| 3 | Customer Churn / Sales Trend Analysis | `project-03-sales-trend-analysis/` | Feature engineering, dates, pivot tables, business recommendations |
| 4 | First Machine Learning Classifier | `project-04-ml-classifier/` | Train/test split, scikit-learn, accuracy, confusion matrix, MSE |
| 5 | Streamlit Interactive ML Web App | `project-05-streamlit-ml-app/` | Streamlit app, joblib model, live prediction |

## How to run

```bash
pip install -r requirements.txt
python project-01-eda-titanic/titanic_eda.py
python project-02-data-visualization/visualization.py
python project-03-sales-trend-analysis/sales_analysis.py
python project-04-ml-classifier/ml_classifier.py
cd project-05-streamlit-ml-app
streamlit run app.py
```

Each project folder has its own README with steps, findings and how to reproduce the results.

## Data sources

- Projects 1, 2, 4 and 5 use the classic Titanic dataset (891 passengers), loaded from the public CSV linked in the InternCircle project guide: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- Project 3 uses simulated e-commerce sales data generated inside the script (seed 42). It is clearly labelled as generated/simulated data in that project's README and notes, not real customer data.

## Notes

- All code was written for this internship and is free to run with the listed open-source libraries.
- No personal data other than the author's name and internship ID is included in this repository.
