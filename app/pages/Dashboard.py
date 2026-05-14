import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Business Dashboard", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] ul { display: none; }
.stApp { background-color: #0d1117; }
body, [class*="css"] { background-color: #0d1117; color: #ffffff; }
section[data-testid="stSidebar"] { background-color: #090d13 !important; }
header[data-testid="stHeader"] { background-color: #0d1117 !important; }
h1, h2, h3, p, label, span, div { color: #ffffff !important; }
hr { border-color: #ffffff !important; opacity: 0.2; }
section[data-testid="stSidebar"] hr { border-color: #ffffff !important; opacity: 0.15; }

/* Fix toolbar */
div[data-testid="stToolbar"] { background-color: #0d1117 !important; }
div[data-testid="stToolbar"] button { color: #ffffff !important; }
div[data-testid="stToolbar"] svg { fill: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("Credit Interactive Dashboard")
        st.markdown("FNB Dataquest 2026")
        st.divider()
        st.markdown("**Navigation**")
        st.page_link("main.py",            label="🏠 Overview")
        st.page_link("pages/EDA.py",       label="📊 EDA Tool")
        st.page_link("pages/Model.py",     label="🎯 Model Evaluation")
        st.page_link("pages/Dashboard.py", label="💰 Business Dashboard")
        st.divider()
        st.markdown("**Model Info**")
        st.metric("AUC", "0.7813", delta="+0.1013")
        st.metric("Gini", "0.5627")
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Train", "84,274")
        col2.metric("Test",  "36,084")

render_sidebar()

st.title("Business Dashboard")