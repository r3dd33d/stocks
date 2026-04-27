import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

DATA_PATH = "data/kairognos_combined_2015_2026_pre_breakout.parquet"


@st.cache_data(max_entries=1)
def load_data() -> pd.DataFrame:
    """Full dataset (all 115 cols, ~4.7M rows). Cached once per session."""
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(max_entries=1)
def load_summary() -> dict:
    """Lightweight summary for the home page: reads only the columns
    needed for headline metrics + a small head sample. Avoids loading
    the full 4 GB dataframe just to render 4 metrics."""
    light = pd.read_parquet(
        DATA_PATH,
        columns=["symbol", "gain_10pct_30d", "timestamp"],
    )
    light["timestamp"] = pd.to_datetime(light["timestamp"], utc=True)
    pf = pq.ParquetFile(DATA_PATH)
    head = pf.read_row_group(0).to_pandas().head(20)
    head["timestamp"] = pd.to_datetime(head["timestamp"], utc=True)
    return {
        "rows": len(light),
        "symbols": int(light["symbol"].nunique()),
        "pos_rate": float(light["gain_10pct_30d"].mean()),
        "n_features": pf.schema_arrow.names.__len__(),
        "head": head,
        "last_date": light["timestamp"].max(),
    }
