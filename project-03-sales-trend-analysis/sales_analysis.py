"""
Project 03 - Customer Churn / Sales Trend Analysis.
InternCircle Data Science & AI Internship - Ritun Kumar
Student ID: IC-2026-E21E85D134E22C70

Run:  python sales_analysis.py
Data: SIMULATED e-commerce sales data, generated inside this script
      (seed 42). This is NOT real customer data. It is used because the
      project allows generated data when it is clearly labelled as such.
      5,000 orders, 1,200 customer IDs (1,177 with at least one order),
      1 Jan 2024 - 31 Dec 2025, in INR.
      A festive-season uplift (Oct-Dec) is built into the generator and is
      documented below, so the monthly trend has a real pattern to find.
Outputs: printed analysis, charts in outputs/ and outputs/sales_summary.md.
"""

import calendar
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)
N_ORDERS = 5000
N_CUSTOMERS = 1200

CATEGORIES = {
    "Electronics": (999, 19999, 0.22),
    "Clothing": (299, 2999, 0.30),
    "Home & Kitchen": (499, 7999, 0.20),
    "Beauty": (199, 1999, 0.15),
    "Sports": (499, 5999, 0.13),
}
REGIONS = ["North", "South", "East", "West"]
REGION_PROBS = [0.30, 0.28, 0.18, 0.24]
# Monthly demand weights (same pattern each year): festive Oct-Dec is strongest.
MONTH_WEIGHTS = [0.70, 0.75, 0.85, 0.90, 0.95, 1.00, 1.00, 1.05, 1.00, 1.15, 1.30, 1.50]

sns.set_theme(style="whitegrid", palette="Set2")


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def generate_data():
    months = [(y, m) for y in (2024, 2025) for m in range(1, 13)]
    weights = np.array(MONTH_WEIGHTS * 2, dtype=float)
    weights = weights / weights.sum()
    month_idx = rng.choice(len(months), size=N_ORDERS, p=weights)
    dates = []
    for i in month_idx:
        year, month = months[i]
        day = int(rng.integers(1, calendar.monthrange(year, month)[1] + 1))
        dates.append(pd.Timestamp(year, month, day))
    cats = list(CATEGORIES.keys())
    probs = [CATEGORIES[c][2] for c in cats]
    category = rng.choice(cats, size=N_ORDERS, p=probs)
    unit_price = np.zeros(N_ORDERS)
    for cat in cats:
        low, high, _ = CATEGORIES[cat]
        mask = category == cat
        unit_price[mask] = rng.uniform(low, high, size=int(mask.sum()))
    df = pd.DataFrame({
        "OrderID": np.arange(1, N_ORDERS + 1),
        "CustomerID": rng.integers(1, N_CUSTOMERS + 1, size=N_ORDERS),
        "OrderDate": dates,
        "Category": category,
        "Region": rng.choice(REGIONS, size=N_ORDERS, p=REGION_PROBS),
        "Quantity": rng.choice([1, 1, 1, 2, 2, 3, 4, 5], size=N_ORDERS),
        "UnitPrice": np.round(unit_price, 2),
        "DiscountPct": rng.choice([0, 0, 5, 10, 15, 20], size=N_ORDERS),
    })
    return df


def main():
    df = generate_data()
    # --- Feature engineering (dates, money, customer behaviour) ---
    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["MonthName"] = df["OrderDate"].dt.month_name()
    df["Quarter"] = df["OrderDate"].dt.quarter
    df["Weekday"] = df["OrderDate"].dt.day_name()
    df["IsWeekend"] = df["OrderDate"].dt.dayofweek >= 5
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").astype(str)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["DiscountAmount"] = df["Revenue"] * df["DiscountPct"] / 100
    df["NetRevenue"] = df["Revenue"] - df["DiscountAmount"]

    total_revenue = df["NetRevenue"].sum()
    print("Orders:", len(df), " Customers:", df["CustomerID"].nunique())
    print("Total net revenue (INR):", round(total_revenue, 2))
    print("Average order value (INR):", round(df["NetRevenue"].mean(), 2))

    print("\n=== Monthly revenue trend ===")
    monthly = df.groupby("YearMonth")["NetRevenue"].sum()
    print(monthly.round(0))
    print("Best month:", monthly.idxmax(), round(monthly.max(), 2))
    print("Weakest month:", monthly.idxmin(), round(monthly.min(), 2))

    print("\n=== Pivot: revenue by category ===")
    by_cat = df.pivot_table(index="Category", values="NetRevenue",
                            aggfunc=["sum", "mean", "count"]).round(2)
    print(by_cat)

    print("\n=== Pivot: revenue by region ===")
    by_region = df.pivot_table(index="Region", values="NetRevenue",
                               aggfunc=["sum", "mean", "count"]).round(2)
    print(by_region)

    print("\n=== Pivot: revenue by category x region ===")
    cat_region = df.pivot_table(index="Category", columns="Region",
                                values="NetRevenue", aggfunc="sum").round(0)
    print(cat_region)

    print("\n=== Pivot: revenue by quarter ===")
    print(df.pivot_table(index=["Year", "Quarter"], values="NetRevenue",
                         aggfunc="sum").round(0))

    # --- Customer segments and churn ---
    cust = df.groupby("CustomerID").agg(
        Orders=("OrderID", "count"),
        TotalSpend=("NetRevenue", "sum"),
        LastOrder=("OrderDate", "max"),
        FirstOrder=("OrderDate", "min"),
    )
    cust["AvgOrderValue"] = cust["TotalSpend"] / cust["Orders"]
    cust["Segment"] = pd.cut(cust["TotalSpend"], bins=[-1, 8000, 25000, 60000, 10**9],
                             labels=["Bronze", "Silver", "Gold", "Platinum"])
    max_date = df["OrderDate"].max()
    cust["RecencyDays"] = (max_date - cust["LastOrder"]).dt.days
    cust["Status"] = pd.cut(cust["RecencyDays"], bins=[-1, 90, 180, 10**6],
                           labels=["Active", "At risk", "Churned"])
    print("\n=== Customers by segment ===")
    print(cust.groupby("Segment", observed=True).agg(
        Customers=("TotalSpend", "size"), Revenue=("TotalSpend", "sum")).round(0))
    print("\n=== Customers by churn status (last order vs", max_date.date(), ") ===")
    print(cust["Status"].value_counts())
    print("Churned share:", round((cust["Status"] == "Churned").mean(), 3))
    print("Repeat customers (2+ orders):", int((cust["Orders"] >= 2).sum()),
          "of", len(cust))

    # --- Charts ---
    fig, ax = plt.subplots(figsize=(11, 5))
    monthly.plot(ax=ax, marker="o", color="#1f4e79")
    ax.set_title("Monthly net revenue trend (simulated data, 2024-2025)")
    ax.set_ylabel("Net revenue (INR)")
    plt.xticks(rotation=45)
    save(fig, "monthly_trend.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("Category")["NetRevenue"].sum().sort_values().plot.barh(ax=ax, color="#2e8b8b")
    ax.set_title("Total net revenue by category")
    ax.set_xlabel("Net revenue (INR)")
    save(fig, "revenue_by_category.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("Region")["NetRevenue"].sum().plot.bar(ax=ax, color=["#1f4e79", "#2e8b8b", "#e67e22", "#8e44ad"])
    ax.set_title("Total net revenue by region")
    ax.set_ylabel("Net revenue (INR)")
    save(fig, "revenue_by_region.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    cust["Status"].value_counts().plot.bar(ax=ax, color=["#27ae60", "#e67e22", "#c0392b"])
    ax.set_title("Customers by churn status")
    ax.set_ylabel("Number of customers")
    save(fig, "churn_status.png")

    # --- Written summary with the exact figures ---
    summary = []
    summary.append("# Sales Trend & Churn Summary (simulated data)")
    summary.append("")
    summary.append("Data: 5,000 simulated orders, 1,200 customer IDs (1,177 with at least one order), Jan 2024-Dec 2025, INR. Generated with seed 42.")
    summary.append("")
    summary.append(f"Total net revenue: INR {total_revenue:,.0f}")
    summary.append(f"Average order value: INR {df['NetRevenue'].mean():,.0f}")
    summary.append(f"Best month: {monthly.idxmax()} (INR {monthly.max():,.0f}); weakest month: {monthly.idxmin()} (INR {monthly.min():,.0f})")
    summary.append(f"Churned customers (no order in last 180 days): {(cust['Status'] == 'Churned').sum()} of {len(cust)} ({(cust['Status'] == 'Churned').mean():.1%})")
    summary.append("")
    summary.append("## Revenue by category")
    summary.append("```")
    summary.append(str(by_cat))
    summary.append("```")
    summary.append("")
    summary.append("## Revenue by region")
    summary.append("```")
    summary.append(str(by_region))
    summary.append("```")
    summary.append("")
    summary.append("## Monthly revenue")
    summary.append("```")
    summary.append(str(monthly.round(0)))
    summary.append("```")
    with open(os.path.join(OUT_DIR, "sales_summary.md"), "w") as f:
        f.write("\n".join(summary))
    print("\nSaved: outputs/sales_summary.md")
    print("\nBusiness recommendations: see business_brief.md in this folder.")


if __name__ == "__main__":
    main()
