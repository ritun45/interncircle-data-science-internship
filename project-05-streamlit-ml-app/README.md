# Project 05 — Streamlit Interactive ML Web App

**InternCircle Data Science & AI Internship — Ritun Kumar (IC-2026-E21E85D134E22C70)**

## Aim

Deploy the trained model from Project 04 as a simple interactive web app: input sliders and selectors, with a real-time prediction output.

## Files

- `app.py` — the Streamlit app.
- `train_model.py` — trains the Decision Tree model and saves it to `models/titanic_survival_model.pkl`.
- `requirements.txt` — Python libraries for the app.
- `models/` — created by `train_model.py` (the saved model file).

## How to run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## How to deploy (free, Streamlit Community Cloud)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Choose "Create app", select this repository (`ritun45/interncircle-data-science-internship`), branch `main`, and main file path `project-05-streamlit-ml-app/app.py`.
3. Deploy. The app loads the saved model if present; if the model file is not in the repo, the app trains the same model itself from the Titanic CSV, so it still works.

## What the app does

- Sidebar inputs: passenger class (1-3), sex, age slider (0-80), siblings/spouses aboard, parents/children aboard, fare slider, and embarkation port.
- On "Predict survival" it shows the survival probability (with a progress bar) and a plain-language prediction, plus a bar chart of which features the model relies on most.
- Model: Decision Tree Classifier (max_depth=5) from Project 04, trained on the classic Titanic dataset (891 passengers).

## Tools used

Python, Streamlit, Scikit-Learn, Pandas, NumPy, joblib.
