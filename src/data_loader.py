import pandas as pd
import streamlit as st

DATA_PATH = "data/kairognos_ml_dataset_with_mktcap.parquet"

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["signal_date"] = pd.to_datetime(df["signal_date"], utc=True)
    df["entry_date"] = pd.to_datetime(df["entry_date"], utc=True)
    df["exit_date"] = pd.to_datetime(df["exit_date"], utc=True)
    return df
