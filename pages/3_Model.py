import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Model", page_icon=":material/model_training:", layout="wide")
st.title(":material/model_training: Model results")

st.caption("Model results will appear here after training. Add evaluation metrics, confusion matrix, and feature importance charts.")
