import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Model", page_icon="🤖", layout="wide")
st.title("🤖 Model Results")

st.info("Model results will appear here after training. Add your evaluation metrics, confusion matrix, and feature importance charts.")
