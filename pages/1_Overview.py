import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data

st.set_page_config(page_title="Overview", page_icon=":material/dataset:", layout="wide")
st.title(":material/dataset: Data Overview")


@st.cache_data(max_entries=1)
def _dtypes_summary():
    df = load_data()
    return df.dtypes.rename("dtype").to_frame()


@st.cache_data(max_entries=1)
def _missing_summary():
    df = load_data()
    missing = df.isnull().sum()
    return missing[missing > 0].sort_values(ascending=False).rename("missing").to_frame()


@st.cache_data(max_entries=1)
def _describe_summary():
    df = load_data()
    return df.describe()


@st.cache_data(max_entries=1)
def _shape():
    df = load_data()
    return len(df), df.shape[1]


rows, cols = _shape()

st.subheader("Shape & types")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Rows:** {rows:,}  |  **Columns:** {cols}")
    st.dataframe(_dtypes_summary(), height=300)
with col2:
    missing_df = _missing_summary()
    st.write(f"**Columns with missing values:** {len(missing_df)}")
    st.dataframe(missing_df, height=300)

st.subheader("Descriptive statistics")
st.dataframe(_describe_summary(), use_container_width=True)
