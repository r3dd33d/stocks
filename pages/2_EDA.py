import streamlit as st
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

st.set_page_config(page_title="EDA", page_icon=":material/insights:", layout="wide")
st.title(":material/insights: Exploratory data analysis")

df = load_data()
LABEL = "gain_10pct_30d"

valid = df.dropna(subset=[LABEL])
overall_pos = valid[LABEL].mean()

# --- Label distribution ---
st.subheader("Label distribution — gain_10pct_30d")
col1, col2 = st.columns(2)
with col1:
    counts = df[LABEL].value_counts(dropna=False).rename({1.0: "Win (≥10%)", 0.0: "No win"})
    st.dataframe(counts.rename("rows").to_frame())
with col2:
    st.metric("Overall positive rate", f"{overall_pos:.2%}")
    st.metric("Total signals", f"{len(df):,}")
    st.metric("Unique symbols", f"{df['symbol'].nunique():,}")

# --- Win rate over time (yearly) ---
st.subheader("Positive rate by year")
yearly = (
    valid.assign(year=valid["timestamp"].dt.year)
    .groupby("year")[LABEL]
    .agg(["mean", "count"])
    .rename(columns={"mean": "pos_rate"})
)
fig, ax = plt.subplots(figsize=(9, 3.5))
yearly["pos_rate"].plot(kind="bar", ax=ax, color="steelblue")
ax.axhline(overall_pos, color="red", linestyle="--", label=f"Overall ({overall_pos:.1%})")
ax.set_ylabel("Pos rate")
ax.set_xlabel("")
ax.legend()
st.pyplot(fig)
st.caption(f"Counts per year: {yearly['count'].to_dict()}")

# --- Win rate by market regime ---
st.subheader("Positive rate by market regime")
if "market_regime" in df.columns:
    by_reg = valid.groupby("market_regime")[LABEL].agg(["mean", "count"]).rename(columns={"mean": "pos_rate"})
    fig, ax = plt.subplots(figsize=(8, 3))
    by_reg["pos_rate"].sort_values(ascending=False).plot(kind="bar", ax=ax, color="mediumseagreen")
    ax.axhline(overall_pos, color="red", linestyle="--")
    ax.set_ylabel("Pos rate")
    st.pyplot(fig)
    st.dataframe(by_reg)

# --- Win rate by exchange ---
st.subheader("Positive rate by exchange")
by_exch = valid.groupby("exchange")[LABEL].agg(["mean", "count"]).rename(columns={"mean": "pos_rate"})
fig, ax = plt.subplots(figsize=(8, 3))
by_exch["pos_rate"].sort_values(ascending=False).plot(kind="bar", ax=ax, color="coral")
ax.axhline(overall_pos, color="red", linestyle="--")
ax.set_ylabel("Pos rate")
st.pyplot(fig)
st.dataframe(by_exch)

# --- Win rate by engine total_score bucket ---
st.subheader("Positive rate by engine `total_score` bucket")
if "total_score" in df.columns:
    sub = valid.dropna(subset=["total_score"]).copy()
    sub["score_bucket"] = pd.cut(sub["total_score"], bins=[-1, 30, 50, 70, 90, 110])
    by_score = sub.groupby("score_bucket", observed=True)[LABEL].agg(["mean", "count"]).rename(columns={"mean": "pos_rate"})
    fig, ax = plt.subplots(figsize=(8, 3.2))
    by_score["pos_rate"].plot(kind="bar", ax=ax, color="purple")
    ax.axhline(overall_pos, color="red", linestyle="--")
    ax.set_ylabel("Pos rate")
    ax.set_xlabel("total_score bucket")
    st.pyplot(fig)
    st.dataframe(by_score)

# --- Feature explorer ---
st.subheader("Feature vs. label")
numeric_cols = [c for c in df.select_dtypes("number").columns if c != LABEL]
default = "rsi_14" if "rsi_14" in numeric_cols else numeric_cols[0]
col = st.selectbox("Pick a numeric feature", numeric_cols, index=numeric_cols.index(default))

sub = valid.dropna(subset=[col]).copy()
lo, hi = sub[col].quantile(0.01), sub[col].quantile(0.99)
sub[col] = sub[col].clip(lo, hi)

# Bin into deciles and show pos rate per bin
sub["bucket"] = pd.qcut(sub[col], q=10, duplicates="drop")
by_feat = sub.groupby("bucket", observed=True)[LABEL].agg(["mean", "count"]).rename(columns={"mean": "pos_rate"})
fig, ax = plt.subplots(figsize=(10, 3.5))
by_feat["pos_rate"].plot(kind="bar", ax=ax, color="steelblue")
ax.axhline(overall_pos, color="red", linestyle="--", label=f"Overall ({overall_pos:.1%})")
ax.set_ylabel("Pos rate")
ax.set_xlabel(f"{col} (decile bucket)")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)
st.caption(f"{col}: clipped to 1–99th percentile. Counts per decile: {by_feat['count'].to_dict()}")

# Distribution split by label
st.subheader(f"Distribution of `{col}` by label")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.hist(sub.loc[sub[LABEL] == 0, col], bins=60, alpha=0.5, label="No win", color="gray", density=True)
ax.hist(sub.loc[sub[LABEL] == 1, col], bins=60, alpha=0.5, label="Win", color="steelblue", density=True)
ax.set_xlabel(col)
ax.set_ylabel("Density")
ax.legend()
st.pyplot(fig)
