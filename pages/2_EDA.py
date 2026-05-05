import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

st.set_page_config(page_title="EDA", page_icon=":material/insights:", layout="wide")
st.title(":material/insights: Exploratory Data Analysis")

# ── Theme ─────────────────────────────────────────────────────────────────────
GREEN_DARK   = "#1B4332"
GREEN_MID    = "#40695B"
GREEN_LIGHT  = "#F0F7F4"
NEGATIVE     = "#E76F51"

FEATURE_GROUPS = {
    "Trend":     ["ma_200", "ma_slope", "price_vs_ma_200_pct", "ath_distance_pct", "days_since_ath"],
    "Momentum":  ["rsi_14", "macd", "macd_histogram", "stoch_k", "adx", "cci_20", "plus_di", "minus_di"],
    "Volatility":["atr_5", "atr_10", "adr", "adr_past", "bb_width_pct", "consolidation_range_pct"],
    "Volume":    ["volume", "avg_volume_30d", "volume_ratio", "obv", "volume_days_above_avg_10"],
    "Pattern":   ["vcp_t_count", "vcp_t1_depth", "vcp_t2_depth", "consolidation_days",
                  "higher_highs", "higher_lows", "lower_highs", "relative_strength",
                  "sector_rs", "spy_distance_from_200", "vcs_score",
                  "pa_quality_score", "pa_body_score", "pa_momentum_score",
                  "pa_close_score", "pa_volume_score"],
}

CATEGORICAL_COLS = [
    "market_regime", "pa_quality_label", "weekly_trend_status", "vcp_state",
    "slope_category", "rs_state", "vcs_label", "macd_state", "stoch_state",
    "adx_state", "cci_state", "obv_state", "volume_surge_label", "rsi_pullback_label",
]

LABEL = "gain_10pct_30d"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def _load():
    return load_data()

@st.cache_data(show_spinner=False)
def _sample(n=100_000):
    df = _load()
    return df.sample(n=min(n, len(df)), random_state=42)

df     = _load()
sample = _sample()
valid  = df.dropna(subset=[LABEL])
overall_pos = valid[LABEL].mean()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "⚖️ Class Balance",
    "📅 Win Rate Over Time",
    "🌍 Market Regime",
    "📊 Distributions",
    "🏷️ Categorical",
    "🔥 Correlations",
    "🔍 Feature Explorer",
    "❓ Missing Values",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CLASS BALANCE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Label Distribution — gain_10pct_30d")
    st.info("**Challenge:** Only 10.4% of rows are positive — an 8.6:1 imbalance. Accuracy alone is misleading; use F1 or AUC-ROC.", icon="⚠️")

    pos = int(df[LABEL].sum())
    neg = len(df) - pos

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive (≥10% gain)", f"{pos:,}", f"{pos/len(df)*100:.1f}%")
    c2.metric("Negative (no gain)",   f"{neg:,}", f"{neg/len(df)*100:.1f}%")
    c3.metric("Imbalance ratio", f"{neg/pos:.1f} : 1")

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig = px.pie(
            values=[pos, neg], names=["Positive (≥10%)", "Negative"],
            color_discrete_sequence=[GREEN_MID, "#D1D5DB"], hole=0.45,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        fig2 = px.bar(
            x=["Negative", "Positive"], y=[neg, pos],
            color=["Negative", "Positive"],
            color_discrete_map={"Negative": "#D1D5DB", "Positive": GREEN_MID},
            text=[f"{neg:,}", f"{pos:,}"],
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", height=300,
                           margin=dict(l=0, r=0, t=20, b=0))
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — WIN RATE OVER TIME
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Positive Rate by Year")

    yearly = (
        valid.assign(year=valid["timestamp"].dt.year)
        .groupby("year")[LABEL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pos_rate", "count": "rows"})
        .reset_index()
    )
    yearly["pos_pct"] = yearly["pos_rate"] * 100

    fig = px.bar(yearly, x="year", y="pos_pct",
                 color_discrete_sequence=[GREEN_MID], text="pos_pct")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_hline(y=overall_pos * 100, line_dash="dash", line_color=NEGATIVE,
                  annotation_text=f"Overall {overall_pos:.1%}")
    fig.update_layout(yaxis_title="Positive rate (%)", xaxis_title="",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      height=340, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Row counts per year"):
        st.dataframe(yearly.rename(columns={"year": "Year", "rows": "Rows",
                                            "pos_pct": "Win Rate (%)"}),
                     use_container_width=True, hide_index=True)

    # By exchange
    st.subheader("Positive Rate by Exchange")
    by_exch = valid.groupby("exchange")[LABEL].agg(["mean", "count"]).reset_index()
    by_exch.columns = ["Exchange", "Win Rate", "Rows"]
    by_exch["Win Rate (%)"] = (by_exch["Win Rate"] * 100).round(2)
    fig2 = px.bar(by_exch.sort_values("Win Rate (%)", ascending=False),
                  x="Exchange", y="Win Rate (%)",
                  color_discrete_sequence=[GREEN_MID], text="Win Rate (%)")
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.add_hline(y=overall_pos * 100, line_dash="dash", line_color=NEGATIVE)
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=280,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MARKET REGIME
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Market Regime Analysis")

    regime_colors = {"Bullish": GREEN_MID, "Neutral": "#94A3B8", "Bearish": NEGATIVE}
    regime_df = valid.groupby("market_regime")[LABEL].agg(["mean", "count"]).reset_index()
    regime_df.columns = ["Regime", "Win Rate", "Count"]
    regime_df["Win Rate (%)"] = (regime_df["Win Rate"] * 100).round(2)
    regime_df["% of Data"] = (regime_df["Count"] / len(df) * 100).round(1)

    for _, row in regime_df.iterrows():
        pass
    c1, c2, c3 = st.columns(3)
    for col_w, (_, row) in zip([c1, c2, c3], regime_df.iterrows()):
        col_w.metric(row["Regime"],
                     f"{row['Win Rate (%)']:.1f}% win rate",
                     f"{row['Count']:,} rows ({row['% of Data']}%)",
                     delta_color="off")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Row count by regime**")
        fig = px.bar(regime_df, x="Regime", y="Count",
                     color="Regime", color_discrete_map=regime_colors, text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", height=300,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("**Win rate by regime**")
        fig2 = px.bar(regime_df, x="Regime", y="Win Rate (%)",
                      color="Regime", color_discrete_map=regime_colors,
                      text="Win Rate (%)")
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.add_hline(y=overall_pos * 100, line_dash="dash", line_color="#374151",
                       annotation_text=f"Overall {overall_pos:.1%}")
        fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", height=300,
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Win rate over time by regime**")
    rt = valid.groupby([valid["timestamp"].dt.year, "market_regime"])[LABEL].mean().reset_index()
    rt.columns = ["Year", "Regime", "Win Rate"]
    fig3 = px.line(rt, x="Year", y="Win Rate", color="Regime",
                   markers=True, color_discrete_map=regime_colors)
    fig3.update_layout(yaxis_tickformat=".0%", plot_bgcolor="rgba(0,0,0,0)",
                       paper_bgcolor="rgba(0,0,0,0)", height=280,
                       margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — NUMERIC DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Numeric Feature Distributions")

    col_sel, col_split = st.columns([2, 1])
    with col_sel:
        selected_group = st.selectbox("Feature group", list(FEATURE_GROUPS.keys()))
    with col_split:
        split_by_label = st.toggle("Split by label (Win / No Win)", value=False)

    cols_to_show = [c for c in FEATURE_GROUPS[selected_group]
                    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not cols_to_show:
        st.warning("No numeric columns found for this group.")
    else:
        n_cols = 3
        rows_grid = [cols_to_show[i:i+n_cols] for i in range(0, len(cols_to_show), n_cols)]
        for row in rows_grid:
            grid = st.columns(n_cols)
            for i, col in enumerate(row):
                with grid[i]:
                    col_data = sample[[col, LABEL]].dropna()
                    if split_by_label:
                        fig = px.histogram(
                            col_data, x=col, color=LABEL, barmode="overlay",
                            opacity=0.7, nbins=40,
                            color_discrete_map={True: GREEN_MID, False: "#D1D5DB"},
                        )
                    else:
                        fig = px.histogram(col_data, x=col, nbins=40,
                                           color_discrete_sequence=[GREEN_MID])
                    fig.update_layout(title=col, title_font_size=12, height=220,
                                      showlegend=split_by_label, bargap=0.05,
                                      margin=dict(l=0, r=0, t=30, b=0),
                                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — CATEGORICAL BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Categorical Feature Breakdown")
    st.caption("Win rate = % of rows where gain_10pct_30d is True")

    cat_avail = [c for c in CATEGORICAL_COLS if c in df.columns]
    selected_cat = st.selectbox("Select column", cat_avail)

    cat_data = df.groupby(selected_cat).agg(
        Count=(LABEL, "count"), Win_Rate=(LABEL, "mean"),
    ).reset_index().sort_values("Count", ascending=False)
    cat_data["Win Rate (%)"] = (cat_data["Win_Rate"] * 100).round(2)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Row count per category**")
        fig = px.bar(cat_data, x="Count", y=selected_cat, orientation="h",
                     color_discrete_sequence=[GREEN_MID], text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=max(250, len(cat_data) * 36), plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=60, t=10, b=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Win rate per category**")
        fig2 = px.bar(cat_data, x="Win Rate (%)", y=selected_cat, orientation="h",
                      color="Win Rate (%)", text="Win Rate (%)",
                      color_continuous_scale=[[0, "#D1D5DB"], [0.5, GREEN_MID], [1, GREEN_DARK]])
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.add_vline(x=overall_pos * 100, line_dash="dash", line_color=NEGATIVE,
                       annotation_text=f"Avg {overall_pos:.1%}")
        fig2.update_layout(height=max(250, len(cat_data) * 36), plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=80, t=10, b=0),
                           coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("View data table"):
        st.dataframe(cat_data.drop(columns="Win_Rate"), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Correlation Matrix")

    selected_grp = st.selectbox("Feature group", list(FEATURE_GROUPS.keys()), key="corr_grp")
    corr_cols = [c for c in FEATURE_GROUPS[selected_grp]
                 if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    corr_cols_with_label = corr_cols + [LABEL]

    if len(corr_cols) < 2:
        st.warning("Not enough numeric columns in this group.")
    else:
        corr_matrix = sample[corr_cols_with_label].dropna().corr()

        fig = px.imshow(
            corr_matrix,
            color_continuous_scale=[[0, NEGATIVE], [0.5, "white"], [1, GREEN_MID]],
            zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
        )
        fig.update_traces(textfont_size=10)
        fig.update_layout(height=max(400, len(corr_cols) * 42),
                          margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top correlations with label**")
        label_corr = corr_matrix[LABEL].drop(LABEL).sort_values(key=abs, ascending=False)
        corr_df = label_corr.reset_index()
        corr_df.columns = ["Feature", "Correlation with gain_10pct_30d"]
        st.dataframe(corr_df.style.background_gradient(
            cmap="RdYlGn", subset=["Correlation with gain_10pct_30d"], vmin=-1, vmax=1
        ), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — FEATURE EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Feature vs. Label Explorer")

    numeric_cols = [c for c in df.select_dtypes("number").columns if c != LABEL]
    default_idx  = numeric_cols.index("rsi_14") if "rsi_14" in numeric_cols else 0
    col_pick     = st.selectbox("Pick a numeric feature", numeric_cols, index=default_idx)

    sub = valid.dropna(subset=[col_pick]).copy()
    lo, hi = sub[col_pick].quantile(0.01), sub[col_pick].quantile(0.99)
    sub[col_pick] = sub[col_pick].clip(lo, hi)
    sub["bucket"] = pd.qcut(sub[col_pick], q=10, duplicates="drop")

    by_feat = (sub.groupby("bucket", observed=True)[LABEL]
               .agg(["mean", "count"])
               .rename(columns={"mean": "Win Rate", "count": "Rows"})
               .reset_index())
    by_feat["bucket_str"] = by_feat["bucket"].astype(str)
    by_feat["Win Rate (%)"] = (by_feat["Win Rate"] * 100).round(2)

    st.markdown(f"**Win rate by decile of `{col_pick}`** (clipped 1–99th pct)")
    fig = px.bar(by_feat, x="bucket_str", y="Win Rate (%)",
                 color_discrete_sequence=[GREEN_MID], text="Win Rate (%)")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_hline(y=overall_pos * 100, line_dash="dash", line_color=NEGATIVE,
                  annotation_text=f"Overall {overall_pos:.1%}")
    fig.update_layout(xaxis_title=f"{col_pick} decile", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=340,
                      margin=dict(l=0, r=0, t=10, b=0))
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Distribution of `{col_pick}` by label**")
    sub_sample = sub.sample(min(50_000, len(sub)), random_state=42)
    fig2 = px.histogram(
        sub_sample, x=col_pick, color=LABEL, barmode="overlay",
        opacity=0.65, nbins=60,
        color_discrete_map={True: GREEN_MID, False: "#D1D5DB"},
        labels={LABEL: "Gained ≥10%"},
    )
    fig2.update_layout(xaxis_title=col_pick, yaxis_title="Count",
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    if "total_score" in df.columns:
        st.subheader("Win Rate by Engine Score Bucket")
        score_sub = valid.dropna(subset=["total_score"]).copy()
        score_sub["score_bucket"] = pd.cut(score_sub["total_score"], bins=[-1, 30, 50, 70, 90, 110])
        by_score = (score_sub.groupby("score_bucket", observed=True)[LABEL]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "Win Rate", "count": "Rows"})
                    .reset_index())
        by_score["bucket_str"] = by_score["score_bucket"].astype(str)
        by_score["Win Rate (%)"] = (by_score["Win Rate"] * 100).round(2)

        fig3 = px.bar(by_score, x="bucket_str", y="Win Rate (%)",
                      color_discrete_sequence=[GREEN_MID], text="Win Rate (%)")
        fig3.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig3.add_hline(y=overall_pos * 100, line_dash="dash", line_color=NEGATIVE)
        fig3.update_layout(xaxis_title="total_score bucket", plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", height=320,
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Missing Values — Root Cause Analysis")
    st.success("**Current status: 0 nulls remaining.** All problematic rows were removed (80,666 rows dropped, 12.3% of original dataset).", icon="✅")

    st.markdown("### What we found before cleaning")

    missing_explanation = pd.DataFrame([
        {
            "Group": "A",
            "Affected Columns": "sector_rs, adr_past, vcp_t1_depth, vcp_t2_depth, vcp_is_valid, sector_leading, stock_leading_sector, distribution_days",
            "# Null Rows": "74,249",
            "% of Data": "11.28%",
            "# Symbols": 95,
            "Root Cause": "No GICS sector classification — ETFs and foreign securities have no sector, so sector_rs and all sector-dependent columns cannot be computed.",
            "Example Symbols": "EWU, EWW, CUBA, FBZ, K, HES, IGT",
        },
        {
            "Group": "B",
            "Affected Columns": "relative_strength, rs_state",
            "# Null Rows": "20,123",
            "% of Data": "3.06%",
            "# Symbols": 73,
            "Root Cause": "Stocks that were acquired, merged, or delisted — corporate events created data gaps that broke the RS computation vs. SPY benchmark.",
            "Example Symbols": "K (acquired by Mars), HES (acquired by Chevron), IGT (merger), JWN (went private)",
        },
        {
            "Group": "C",
            "Affected Columns": "ma_200, rsi_14, macd, macd_histogram, stoch_k, adx, atr_5, atr_10, cci_20, price_vs_ma_200_pct, and label states",
            "# Null Rows": "6,416",
            "% of Data": "0.97%",
            "# Symbols": 9,
            "Root Cause": "ETFs with insufficient price history — MA200 requires 200 bars of data. When MA200 fails, all dependent indicators also return null.",
            "Example Symbols": "MDY, MINT, MGK, HEZU, MDYG, MDYV, MGV, MIDU, MFEM",
        },
    ])

    st.dataframe(missing_explanation, use_container_width=True, hide_index=True)

    st.markdown("### Decision")
    st.markdown("""
    All rows containing **any null value** were removed:

    | | Rows | Positive Rate |
    |---|---|---|
    | **Before cleaning** | 658,066 | 12.2% |
    | **After cleaning** | 577,400 | 10.4% |
    | **Removed** | 80,666 (12.3%) | — |

    The missingness is **structural** (not random) — caused by specific securities and corporate events —
    so dropping rows is the correct approach. Imputation would introduce false signal.
    """)
