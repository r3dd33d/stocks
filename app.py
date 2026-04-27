import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.data_loader import load_summary

st.set_page_config(
    page_title="Stock Signals ML",
    page_icon=":material/trending_up:",
    layout="wide",
)

st.title(":material/trending_up: Stock Trading Signals – ML Project")
st.markdown("Use the sidebar to navigate between pages.")

with st.spinner("Loading summary..."):
    s = load_summary()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total signals", f"{s['rows']:,}")
col2.metric("Unique symbols", f"{s['symbols']:,}")
col3.metric("Positive rate", f"{s['pos_rate']:.1%}")
col4.metric("Features", s["n_features"])

st.caption(f"Last bar in dataset: {s['last_date'].date()}")
st.dataframe(s["head"], use_container_width=True)
