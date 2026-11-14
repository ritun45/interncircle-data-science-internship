# Project 03 — Customer Churn / Sales Trend Analysis

**InternCircle Data Science & AI Internship — Ritun Kumar (IC-2026-E21E85D134E22C70)**

## Aim

Perform an end-to-end analysis of an e-commerce sales dataset: find the top customer segments and revenue trends, and turn them into business recommendations.

## Important: the data is simulated

No real customer data was available for this project, so the script generates a simulated dataset (seed 42): 5,000 orders from 1,200 customer IDs (1,177 of them placed at least one order), 1 January 2024 to 31 December 2025, in INR. The generator, its category price ranges, region mix and festive-season uplift (October-December) are all documented in `sales_analysis.py`. The data is clearly labelled as generated/simulated in the script, in `business_brief.md`, and in the InternCircle submission notes. All findings below are findings about this simulated dataset, not about a real business.

## Files

- `sales_analysis.py` — generates the data, engineers features, builds pivot tables, analyses churn and writes the summary.
- `business_brief.md` — one-page business brief with recommendations.
- `outputs/` — created when the script runs: 4 charts and `sales_summary.md` with the exact figures.

## How to run

```bash
pip install pandas numpy matplotlib seaborn
python sales_analysis.py
```

## Steps in the script

1. Generate 5,000 simulated orders with order date, category, region, quantity, unit price and discount.
2. Feature engineering: extract Year, Month, Quarter, Weekday and IsWeekend from the order date; compute Revenue, DiscountAmount and NetRevenue.
3. Build pivot tables: monthly revenue trend, revenue by category, by region, by category x region, and by quarter.
4. Segment customers by lifetime spend (Bronze < INR 8,000; Silver INR 8,000-25,000; Gold INR 25,000-60,000; Platinum >= INR 60,000).
5. Churn analysis: a customer is Active if their last order was within 90 days of the latest order date, At risk at 91-180 days, and Churned after 180 days with no order.
6. Save charts and a written summary (`outputs/sales_summary.md`) with the exact totals, best/worst months and churn share.

## Findings (verified by running the script)

- **Totals:** 5,000 orders, 1,177 customers with orders, total net revenue INR 45,058,908.70, average order value INR 9,011.78.
- **Trend:** best month December 2024 (INR 2,733,896.57); weakest month February 2025 (INR 1,361,344.15). 2024 total INR 22,872,946.67; 2025 total INR 22,185,962.03.
- **By category:** Electronics INR 23,647,432.74 (1,093 orders); Home & Kitchen INR 9,741,042.89 (1,044); Clothing INR 5,441,767.64 (1,485); Sports INR 4,416,852.88 (639); Beauty INR 1,811,812.55 (739).
- **By region:** North INR 13,974,669.53 (1,496 orders); South INR 12,620,538.69 (1,400); West INR 10,710,722.96 (1,228); East INR 7,752,977.52 (876).
- **By segment:** Platinum 236 customers / INR 20,710,369; Gold 443 / INR 17,732,743; Silver 379 / INR 6,109,915; Bronze 119 / INR 505,882.
- **Churn:** Active 578, At risk 239, Churned 360 (30.6% of customers). Repeat customers (2+ orders): 1,095 of 1,177.

## Business recommendations

See `business_brief.md`: protect the festive-season peak, win back churned customers, grow the top segment, and fix the weakest region/category combinations shown in the pivot tables.

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn.
