import streamlit as st
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("📊 Data Overview")

df = load_data()

st.subheader("Shape & Types")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Rows:** {len(df):,}  |  **Columns:** {df.shape[1]}")
    st.dataframe(df.dtypes.rename("dtype").to_frame(), height=300)
with col2:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.write(f"**Columns with missing values:** {len(missing)}")
    st.dataframe(missing.rename("missing").to_frame(), height=300)

st.subheader("Descriptive Statistics")
st.dataframe(df.describe(), use_container_width=True)
