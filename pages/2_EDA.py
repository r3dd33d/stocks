import streamlit as st
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")
st.title("🔍 Exploratory Data Analysis")

df = load_data()

# --- Win rate by signal type ---
st.subheader("Win Rate by Signal Type")
win_by_type = df.groupby("sig_signal_type")["won"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
win_by_type.plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Win Rate")
ax.set_xlabel("")
ax.axhline(df["won"].mean(), color="red", linestyle="--", label="Overall avg")
ax.legend()
st.pyplot(fig)

# --- PnL distribution ---
st.subheader("PnL % Distribution")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    df["pnl_pct"].clip(-50, 100).hist(bins=60, ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("PnL %")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    df.groupby("exit_reason")["pnl_pct"].mean().sort_values().plot(kind="barh", ax=ax, color="coral")
    ax.set_title("Avg PnL by Exit Reason")
    st.pyplot(fig)

# --- Win rate by market cap ---
st.subheader("Win Rate by Market Cap Bucket")
win_by_cap = df.groupby("market_cap_bucket")["won"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 3))
win_by_cap.plot(kind="bar", ax=ax, color="mediumseagreen")
ax.set_ylabel("Win Rate")
st.pyplot(fig)

# --- Feature explorer ---
st.subheader("Feature vs. PnL")
numeric_cols = df.select_dtypes("number").columns.tolist()
col = st.selectbox("Select a feature", numeric_cols, index=numeric_cols.index("ind_rsi_14") if "ind_rsi_14" in numeric_cols else 0)
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99)),
           df["pnl_pct"].clip(-50, 100), alpha=0.1, s=2)
ax.set_xlabel(col)
ax.set_ylabel("PnL %")
st.pyplot(fig)
