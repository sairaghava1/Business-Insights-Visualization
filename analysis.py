"""
Business Insights & Revenue Visualization
==========================================
Dataset: Superstore Sales Dataset (Finance/Revenue)
Download from: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
Save as 'superstore.csv' in the same folder as this script.

Run: python analysis.py
Outputs: 6 chart images saved to /charts/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

BLUE   = "#2563EB"; GREEN  = "#10B981"; RED    = "#EF4444"
ORANGE = "#F59E0B"; PURPLE = "#8B5CF6"; GRAY   = "#64748B"; BG = "#F8FAFC"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#CBD5E1",
    "font.family": "DejaVu Sans", "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11,
})

# ── LOAD & CLEAN ──────────────────────────────────────────────────────────────
print("Loading dataset...")
try:
    df = pd.read_csv("superstore.csv", encoding="latin1")
except:
    df = pd.read_csv("superstore.csv", encoding="utf-8")

# Normalise column names
df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
print(f"✓ Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

# Map common column name variants
col_map = {}
for c in df.columns:
    if "order" in c and "date" in c: col_map["order_date"] = c
    if "sale" in c or "revenue" in c: col_map["sales"] = c
    if "profit" in c: col_map["profit"] = c
    if "categor" in c: col_map["category"] = c
    if "sub" in c and "cat" in c: col_map["sub_category"] = c
    if "region" in c: col_map["region"] = c
    if "segment" in c: col_map["segment"] = c
    if "discount" in c: col_map["discount"] = c
    if "quantit" in c: col_map["quantity"] = c

df.rename(columns={v:k for k,v in col_map.items()}, inplace=True)
df["order_date"] = pd.to_datetime(df.get("order_date", df.iloc[:,0]), errors="coerce")
df["year"]  = df["order_date"].dt.year
df["month"] = df["order_date"].dt.to_period("M")
df["profit_margin"] = (df["profit"] / df["sales"] * 100).round(2)

print(f"✓ Date range: {df['order_date'].min().date()} → {df['order_date'].max().date()}")
print(f"  Total Revenue : ${df['sales'].sum():,.0f}")
print(f"  Total Profit  : ${df['profit'].sum():,.0f}")
print(f"  Avg Margin    : {df['profit_margin'].mean():.1f}%")

# ── CHART 1: Revenue & Profit Over Time ───────────────────────────────────────
print("\nGenerating Chart 1: Revenue & Profit Over Time...")
monthly = df.groupby("month").agg(revenue=("sales","sum"), profit=("profit","sum")).reset_index()
monthly["month_dt"] = monthly["month"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.fill_between(monthly["month_dt"], monthly["revenue"]/1e3, alpha=0.15, color=BLUE)
ax.plot(monthly["month_dt"], monthly["revenue"]/1e3, color=BLUE, linewidth=2, label="Revenue ($K)")
ax2 = ax.twinx()
ax2.plot(monthly["month_dt"], monthly["profit"]/1e3, color=GREEN, linewidth=2,
         linestyle="--", label="Profit ($K)")
ax.set_ylabel("Revenue ($K)", color=BLUE)
ax2.set_ylabel("Profit ($K)", color=GREEN)
ax.set_title("Monthly Revenue & Profit Trend", pad=15)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))
ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/01_revenue_profit_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved 01_revenue_profit_trend.png")

# ── CHART 2: Revenue by Category & Sub-Category ───────────────────────────────
print("Generating Chart 2: Revenue by Category...")
cat_stats = df.groupby("category").agg(
    revenue=("sales","sum"), profit=("profit","sum")).reset_index()
cat_stats["margin"] = (cat_stats["profit"]/cat_stats["revenue"]*100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
colors_cat = [BLUE, GREEN, ORANGE][:len(cat_stats)]
bars = axes[0].bar(cat_stats["category"], cat_stats["revenue"]/1e3,
                   color=colors_cat, alpha=0.85, width=0.5)
axes[0].bar_label(bars, fmt=lambda x: f"${x:.0f}K", padding=4, fontsize=10, fontweight="bold")
axes[0].set_ylabel("Revenue ($K)"); axes[0].set_title("Revenue by Category")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))

# Sub-category
subcat = df.groupby("sub_category")["sales"].sum().nlargest(10).reset_index().iloc[::-1]
axes[1].barh(subcat["sub_category"], subcat["sales"]/1e3, color=BLUE, alpha=0.75, height=0.6)
axes[1].set_xlabel("Revenue ($K)"); axes[1].set_title("Top 10 Sub-Categories by Revenue")
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))

plt.suptitle("Revenue Breakdown by Product Category", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/02_revenue_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved 02_revenue_by_category.png")

# ── CHART 3: Regional Performance ─────────────────────────────────────────────
print("Generating Chart 3: Regional Performance...")
region_stats = df.groupby("region").agg(
    revenue=("sales","sum"), profit=("profit","sum"),
    orders=("sales","count")).reset_index()
region_stats["margin"] = (region_stats["profit"]/region_stats["revenue"]*100).round(1)
region_stats = region_stats.sort_values("revenue", ascending=False)

x = np.arange(len(region_stats)); width = 0.38
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
b1 = ax.bar(x - width/2, region_stats["revenue"]/1e3, width, color=BLUE, alpha=0.85, label="Revenue")
ax2 = ax.twinx()
b2 = ax2.bar(x + width/2, region_stats["profit"]/1e3, width, color=GREEN, alpha=0.75, label="Profit")
ax.set_xticks(x); ax.set_xticklabels(region_stats["region"])
ax.set_ylabel("Revenue ($K)", color=BLUE)
ax2.set_ylabel("Profit ($K)", color=GREEN)
ax.set_title("Revenue & Profit by Region", pad=15)
ax.legend(loc="upper left", fontsize=9); ax2.legend(loc="upper right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:.0f}K"))
ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/03_regional_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved 03_regional_performance.png")

# ── CHART 4: Discount vs Profit Analysis ──────────────────────────────────────
print("Generating Chart 4: Discount vs Profit...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

sample = df.sample(min(3000, len(df)), random_state=42)
colors_scatter = [GREEN if p > 0 else RED for p in sample["profit"]]
axes[0].scatter(sample["discount"], sample["profit"], c=colors_scatter,
                alpha=0.4, s=12)
axes[0].axhline(0, color=GRAY, linestyle="--", linewidth=1)
axes[0].axvline(0.3, color=ORANGE, linestyle="--", linewidth=1.2,
                label="30% discount threshold")
axes[0].set_xlabel("Discount Rate"); axes[0].set_ylabel("Profit ($)")
axes[0].set_title("Discount Rate vs Profit\n(red = loss-making orders)")
axes[0].legend(fontsize=8)

discount_bins = pd.cut(df["discount"], bins=[0,0.1,0.2,0.3,0.4,0.5,1.0],
                       labels=["0-10%","10-20%","20-30%","30-40%","40-50%","50%+"])
disc_profit = df.groupby(discount_bins)["profit"].mean().reset_index()
colors_dp = [GREEN if v > 0 else RED for v in disc_profit["profit"]]
axes[1].bar(disc_profit["discount"].astype(str), disc_profit["profit"],
            color=colors_dp, alpha=0.85)
axes[1].axhline(0, color=GRAY, linestyle="--")
axes[1].set_xlabel("Discount Range"); axes[1].set_ylabel("Average Profit ($)")
axes[1].set_title("Average Profit by Discount Range")

plt.suptitle("Impact of Discounting on Profitability", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/04_discount_vs_profit.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved 04_discount_vs_profit.png")

# ── CHART 5: Customer Segment Analysis ────────────────────────────────────────
print("Generating Chart 5: Segment Analysis...")
seg_stats = df.groupby("segment").agg(
    revenue=("sales","sum"), profit=("profit","sum"),
    orders=("sales","count")).reset_index()
seg_stats["avg_order"] = seg_stats["revenue"] / seg_stats["orders"]
seg_stats["margin"]    = (seg_stats["profit"] / seg_stats["revenue"] * 100).round(1)

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.patch.set_facecolor(BG)
colors_seg = [BLUE, GREEN, ORANGE]

for ax_i, (metric, label) in enumerate(zip(
    ["revenue","profit","avg_order"],
    ["Total Revenue ($K)","Total Profit ($K)","Avg Order Value ($)"])):
    vals = seg_stats[metric] / (1e3 if metric != "avg_order" else 1)
    bars = axes[ax_i].bar(seg_stats["segment"], vals, color=colors_seg, alpha=0.85, width=0.5)
    if metric != "avg_order":
        axes[ax_i].bar_label(bars, fmt=lambda x: f"${x:.0f}K", padding=3, fontsize=9)
    else:
        axes[ax_i].bar_label(bars, fmt=lambda x: f"${x:.0f}", padding=3, fontsize=9)
    axes[ax_i].set_title(label)
    axes[ax_i].tick_params(axis="x", rotation=15)

plt.suptitle("Performance by Customer Segment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/05_segment_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved 05_segment_analysis.png")

# ── CHART 6: YoY Revenue Growth & Profit Margin Heatmap ──────────────────────
print("Generating Chart 6: YoY Heatmap...")
pivot = df.groupby(["year","category"])["sales"].sum().unstack().fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

sns.heatmap(pivot/1e3, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.5, ax=axes[0], cbar_kws={"label":"Revenue ($K)"})
axes[0].set_title("Revenue Heatmap\nYear × Category ($K)")
axes[0].set_ylabel("Year"); axes[0].set_xlabel("")

pivot_margin = df.groupby(["year","category"])["profit_margin"].mean().unstack()
sns.heatmap(pivot_margin, annot=True, fmt=".1f", cmap="RdYlGn",
            center=0, linewidths=0.5, ax=axes[1], cbar_kws={"label":"Profit Margin %"})
axes[1].set_title("Profit Margin Heatmap\nYear × Category (%)")
axes[1].set_ylabel(""); axes[1].set_xlabel("")

plt.suptitle("Year-over-Year Business Performance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHARTS_DIR}/06_yoy_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved 06_yoy_heatmap.png")

# ── SQL QUERIES FILE ──────────────────────────────────────────────────────────
sql = """-- ============================================================
-- Business Insights & Revenue Analytics – SQL Queries
-- Dataset: Superstore Sales
-- ============================================================

-- 1. Total Revenue, Profit and Margin by Category
SELECT category,
       ROUND(SUM(sales), 2)                          AS total_revenue,
       ROUND(SUM(profit), 2)                         AS total_profit,
       ROUND(SUM(profit) / SUM(sales) * 100, 2)      AS profit_margin_pct
FROM superstore
GROUP BY category
ORDER BY total_revenue DESC;

-- 2. Monthly Revenue Trend
SELECT DATE_FORMAT(order_date, '%Y-%m')              AS month,
       ROUND(SUM(sales), 2)                          AS monthly_revenue,
       ROUND(SUM(profit), 2)                         AS monthly_profit
FROM superstore
GROUP BY month
ORDER BY month;

-- 3. Top 10 Most Profitable Sub-Categories
SELECT sub_category,
       ROUND(SUM(sales), 2)                          AS revenue,
       ROUND(SUM(profit), 2)                         AS profit,
       ROUND(SUM(profit) / SUM(sales) * 100, 2)      AS margin_pct
FROM superstore
GROUP BY sub_category
ORDER BY profit DESC
LIMIT 10;

-- 4. Loss-Making Orders (discount > 30%)
SELECT order_id, product_name, sales, discount, profit
FROM superstore
WHERE discount >= 0.3 AND profit < 0
ORDER BY profit ASC
LIMIT 20;

-- 5. Revenue & Profit by Region and Segment
SELECT region, segment,
       ROUND(SUM(sales), 2)                          AS revenue,
       ROUND(SUM(profit), 2)                         AS profit,
       COUNT(DISTINCT order_id)                      AS order_count
FROM superstore
GROUP BY region, segment
ORDER BY revenue DESC;

-- 6. Year-over-Year Revenue Growth
SELECT year,
       ROUND(SUM(sales), 2)                          AS revenue,
       ROUND(SUM(profit), 2)                         AS profit,
       ROUND(SUM(profit) / SUM(sales) * 100, 2)      AS margin_pct
FROM (
    SELECT *, YEAR(order_date) AS year FROM superstore
) t
GROUP BY year
ORDER BY year;
"""
with open("queries.sql", "w") as f:
    f.write(sql)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  BUSINESS INSIGHTS SUMMARY")
print("="*55)
print(f"  Total Records    : {len(df):,}")
print(f"  Total Revenue    : ${df['sales'].sum():,.0f}")
print(f"  Total Profit     : ${df['profit'].sum():,.0f}")
print(f"  Avg Profit Margin: {df['profit_margin'].mean():.1f}%")
print(f"  Date Range       : {df['order_date'].min().date()} → {df['order_date'].max().date()}")
print(f"  Regions          : {df['region'].nunique()}")
print(f"  Categories       : {df['category'].nunique()}")
print("="*55)
print(f"\n✅ All charts saved to /{CHARTS_DIR}/")
print(f"✅ SQL queries saved to queries.sql")
