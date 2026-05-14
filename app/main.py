# 0 is no default(low risk), 1 is default(high risk)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

 
st.set_page_config(page_title="Credit Default Interactive Dashboard", layout="wide")

st.title("Credit Default Interactive Dashboard")

from utils import render_sidebar
render_sidebar()

#create sidebar

def render_sidebar():
    with st.sidebar:
        st.markdown("Credit Interactive Dashboard")
        st.markdown("FNB Dataquest 2026")
        st.divider()

        st.markdown("**Navigation**")
        st.page_link("main.py", label="Home")
        st.page_link("pages/1_EDA.py", label="EDA Tool")
        st.page_link("pages/2_Model.py", label="Model evaluation")
        st.page_link("pages/3_Dashboard.py", label="Business Dashboard")

        st.divider()
        st.markdown("**Model Info**")
        st.metric("AUC", "0.7813", delta="+0.1013")
        st.metric("Gini", "0.5627")
        
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Train", "84,274")
        col2.metric("Test", "36,084")





