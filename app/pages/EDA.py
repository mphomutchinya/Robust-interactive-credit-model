import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

df_engineered = pd.read_csv("C:/Users/mutch_lf652j0/Credit Score Interactive Model/data/processed/engineered_train.csv")

st.set_page_config(page_title="EDA Tool", layout="wide")

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


st.title("Exploratory Data Analysis Tool")
st.markdown("Explore the relationship between features and their effects on defaulting")

tabs1, tabs2, tabs3, tab4 = st.tabs(["Univariate", "Bivariate", "Default Rate", "Explore (Raw Data)"])

features = ['age',
 'employment_length_years',
 'num_open_accounts',
 'credit_utilisation_pct',
 'interest_rate',
 'months_since_last_delinquency',
 'pct_accounts_current',
 'missing_annual_income',
 'missing_months_since_last_delinquency',
 'credit_history_to_age',
 'hard_inquiries_delinquencies',
 'rate_dti_burden',
 'rate_to_age',
 'log_annual_income',
 'log_loan_amount',
 'log_total_revolving_balance',
 'log_interest_to_income',
 'log_num_hard_inquiries_6mo',
 'log_num_delinquencies_2yr']

with tabs1:

    left_col, right_col = st.columns(2, border = True)

    with left_col:

        selected_feature = st.selectbox("Select a feature", features)
        plot_type = st.selectbox("Choose plot type", ["Histogram", "KDE plot", "Box plot", "Pie chart"])

    with right_col:

        fig, ax = plt.subplots(figsize = (6,4))

        #Histogram
        if plot_type == "Histogram":
            sns.histplot(df_engineered[features], kde = True, color = "#2563EB")

