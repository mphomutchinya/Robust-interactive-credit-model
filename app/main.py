# 0 is no default(low risk), 1 is default(high risk)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

 
st.set_page_config(page_title="Credit Default Interactive Dashboard", layout="wide")

st.title("Credit Default Interactive Dashboard")

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





