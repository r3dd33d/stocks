import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.data_loader import load_data

st.set_page_config(
    page_title="Stock Signals ML",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Trading Signals – ML Project")
st.markdown("Use the sidebar to navigate between pages.")

with st.spinner("Loading data..."):
    df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Signals", f"{len(df):,}")
col2.metric("Unique Symbols", df["symbol"].nunique())
col3.metric("Win Rate", f"{df['won'].mean():.1%}")
col4.metric("Features", df.shape[1])

st.dataframe(df.head(20), use_container_width=True)
