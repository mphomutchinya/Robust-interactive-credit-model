import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from joblib import load
import pickle
import os

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "logistic_regression_v2.pkl")
model = load(model_path)

df_cleaned = pd.read_csv("C:/Users/mutch_lf652j0/Credit Score Interactive Model/data/processed/cleaned_train.csv")

st.set_page_config(page_title="EDA Tool", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] ul { display: none; }
.stApp { background-color: #0B0F14; }
/*General text*/
body, [class*="css"] { background-color: ##0B0F14; color: #FFFFFF; }
/*Sidebar*/
section[data-testid="stSidebar"] { background-color: #111827 !important; }
/*Header*/
header[data-testid="stHeader"] { background-color: #0B0F14 !important; }
h1, h2, h3, p, label, span, div { color: #ffffff !important; }
hr { border-color: #ffffff !important; opacity: 0.2; }
section[data-testid="stSidebar"] hr { border-color: #ffffff !important; opacity: 0.15; }
/* Titles */
h1, h2, h3 { color: #F58220 !important;
    font-weight: 700;}
/* Paragraphs & Labels */
p, label, span, div { color: #FFFFFF !important;}
/* Horizontal Lines */
hr {border-color: #F58220 !important; opacity: 0.2;}

/* Toolbar */
div[data-testid="stToolbar"] {background-color: #0B0F14 !important;}
div[data-testid="stToolbar"] button {color: #FFFFFF !important;}   
div[data-testid="stToolbar"] svg {
    fill: #FFFFFF !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #1F2937 !important;
    border: 1px solid #F58220 !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background-color: #F58220 !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #F58220;
    padding: 15px;
    border-radius: 12px;
}

.stButton > button:hover {
    background-color: #ff9a3c !important;
    color: white !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: white !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #F58220 !important;
    border-bottom: 3px solid #F58220 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #F58220;
    border-radius: 10px;
}
/*Column containers*/
div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
    border: 2px solid rgba(255,255,255,0.75);
    border-radius: 18px;
    padding: 20px;
    background: rgba(17, 24, 39, 0.45);
    box-shadow: 0 0 12px rgba(255,255,255,0.08);
}
/* Stronger hover glow */
div[data-testid="stVerticalBlock"] > div:has(div.element-container):hover {
    border: 2px solid #FFFFFF;
    box-shadow: 0 0 18px rgba(255,255,255,0.18);
    transition: 0.3s ease;
}
/*Select boxes*/
div[data-baseweb="select"] > div {
    background-color: #1F2937 !important;
    border: 1px solid #F58220 !important;
    border-radius: 12px !important;
    color: white !important;
}
/*Tab styling*/
button[data-baseweb="tab"] {
    color: white !important;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #F58220 !important;
    border-bottom: 3px solid #F58220 !important;
} 
/*Plot container*/
[data-testid="stPlotlyChart"],
[data-testid="stImage"],
[data-testid="stPyplot"] {

    border: 2px solid rgba(255,255,255,0.8);
    border-radius: 18px;
    padding: 12px;
    background-color: #FFFFFF;
}
            /* Dropdown menu background */
div[role="listbox"] {
    background-color: #111827 !important;
    border: 1px solid #F58220 !important;
    color: white !important;
}

/* Dropdown options */
div[role="option"] {
    background-color: #111827 !important;
    color: white !important;
    font-weight: 500;
}

/* Hovered option */
div[role="option"]:hover {
    background-color: #F58220 !important;
    color: white !important;
}

/* Selected option */
div[aria-selected="true"] {
    background-color: rgba(245, 130, 32, 0.25) !important;
    color: white !important;
}

/* Fix dropdown text visibility */
span[data-testid="stMarkdownContainer"] p {
    color: white !important;
}

/* Selectbox arrow */
svg {
    fill: white !important;
}
/* Dropdown popup portal (renders outside normal DOM) */
ul[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #111827 !important;
    border: 1px solid #F58220 !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #ffffff !important;
    background-color: #111827 !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background-color: #F58220 !important;
    color: #ffffff !important;
}

/* Also target the popover container Streamlit uses */
div[data-testid="stPopover"],
div[class*="menuList"],
div[class*="menu"] {
    background-color: #111827 !important;
    color: #ffffff !important;
}
/* Dropdown popup portal (renders outside normal DOM) */
ul[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #111827 !important;
    border: 1px solid #F58220 !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #ffffff !important;
    background-color: #111827 !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background-color: #F58220 !important;
    color: #ffffff !important;
}

/* Also target the popover container Streamlit uses */
div[data-testid="stPopover"],
div[class*="menuList"],
div[class*="menu"] {
    background-color: #111827 !important;
    color: #ffffff !important;
}
            /* Target the floating menu portal by class */
[class*="stMainMenu"],
[class*="main-menu"],
ul[class*="menu"],
div[role="menu"] {
    background-color: #111827 !important;
    border: 1px solid #F58220 !important;
}

div[role="menu"] li,
div[role="menu"] span,
div[role="menu"] a {
    color: #ffffff !important;
    background-color: #111827 !important;
}

div[role="menuitem"]:hover {
    background-color: #F58220 !important;
}

/* Nuclear option - catch any floating white popover */
[data-radix-popper-content-wrapper] > div {
    background-color: #111827 !important;
    border: 1px solid #F58220 !important;
    color: #ffffff !important;
}

[data-radix-popper-content-wrapper] span,
[data-radix-popper-content-wrapper] li,
[data-radix-popper-content-wrapper] a {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("Credit Interactive Dashboard")
        st.markdown("FNB Dataquest 2026")
        st.divider()
        st.markdown("**Navigation**")
        st.page_link("main.py",            label="Overview")
        st.page_link("pages/EDA.py",       label="EDA Tool")
        st.page_link("pages/PD_NewClient.py", label= "Probability of Default New Client")
        st.page_link("pages/Model.py",     label="Model Evaluation")
        st.page_link("pages/Dashboard.py", label="Business Dashboard")
        st.divider()
        st.markdown("**Model Info**")
        st.metric("AUC", "0.7867", delta="+0.1867")
        st.metric("Gini", "0.5733")
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Train", "84 274")
        col2.metric("Test",  "36 084")

render_sidebar()

st.title("Probability of Default")
st.markdown("Enter client details to predict the Probability of Default")

colu1, colu2, colu3 = st.columns(3, border = True)

with colu1:
    st.markdown("#### Personal Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    
    no_income_data = st.checkbox("Annual income unknown", key="income_checkbox")
    
    if no_income_data:
        df_cleaned["emp_bin"] = pd.cut(df_cleaned["employment_length_years"], bins=5)
        emp_bin = pd.cut(
            [employment_length],
            bins=pd.cut(df_cleaned["employment_length_years"], bins=5).cat.categories,
            include_lowest=True
        )[0]
        group_median = df_cleaned.groupby("emp_bin", observed=True)["annual_income"].median()
        annual_income = float(group_median[emp_bin])
        missing_annual_income = 1
    else:
        annual_income = st.number_input("Annual Income", min_value=0.0, value=20000.0)
        missing_annual_income = 0
    
with colu2:
    st.markdown("#### Loan Details")
    loan_amount = st.number_input("Loan Amount", min_value = 0, value = 10000)
    interest_rate = st.number_input("Interest Rate (%)", min_value = 0, value = 10)
    dti_ratio = st.number_input("Debt to income Ratio", min_value = 0.0, value = 0.6)

with colu3:
    st.markdown("#### Credit History")
    num_open_accounts = st.number_input("Number of Open Accounts", min_value = 0, value = 5 )
    credit_utilisation= st.number_input("Credit Utilisation(%)", min_value = 0, max_value = 100, value = 20)
    pct_accounts_current = st.number_input("% Accounts Current", min_value=0.0, max_value=100.0, value=95.0)
    total_revolving_balance = st.number_input("Total Revolving Balance", min_value=0, value=5000)
    num_hard_inquiries = st.number_input("Hard Inquiries (6mo)", min_value=0, value=1)
    num_delinquencies = st.number_input("Delinquencies (2yr)", min_value=0, value=0)
    months_since_oldest_account = st.number_input("Months Since Oldest Account", min_value=0, value=120)
    no_delinquency_history = st.checkbox("No delinquency history")
    if no_delinquency_history:
        months_since_last_delinquency = 900
        missing_months_since_last_delinquency = 1
    else:
        months_since_last_delinquency = st.number_input("Months Since Last Delinquency", min_value=0, value=12)
        missing_months_since_last_delinquency = 0

st.divider()

calculatepd = st.button("Calculate probability of Default")

if calculatepd:

    #engineered features in training model
    credit_history_to_age = months_since_oldest_account / (age *12)
    hard_inquiries_delinquencies_raw = num_hard_inquiries * num_delinquencies
    hard_inquiries_delinquencies = pd.cut(
        [hard_inquiries_delinquencies_raw],
        bins=[-1, 0, 2, 6, float('inf')],
        labels=[0, 1, 2, 3]
    )[0]

    hard_inquiries_delinquencies = int(hard_inquiries_delinquencies)
    rate_dti_burden = interest_rate * dti_ratio
    rate_to_age = interest_rate * (age + 1)

    #log transforms
    log_annual_income = np.log1p(annual_income)
    log_loan_amount             = np.log1p(loan_amount)
    log_total_revolving_balance = np.log1p(total_revolving_balance)
    log_interest_to_income      = np.log1p(interest_rate / (annual_income + 1))
    log_num_hard_inquiries_6mo  = np.log1p(num_hard_inquiries)
    log_num_delinquencies_2yr   = np.log1p(num_delinquencies)


    input_df = pd.DataFrame([{"age": age,
        "employment_length_years": employment_length,
        "num_open_accounts": num_open_accounts,
        "credit_utilisation_pct": credit_utilisation,
        "interest_rate": interest_rate,
        "months_since_last_delinquency": months_since_last_delinquency,
        "pct_accounts_current": pct_accounts_current,
        "missing_annual_income": missing_annual_income,
        "missing_months_since_last_delinquency": missing_months_since_last_delinquency,
        "credit_history_to_age": credit_history_to_age,
        "hard_inquiries_delinquencies": hard_inquiries_delinquencies,
        "rate_dti_burden": rate_dti_burden,
        "rate_to_age": rate_to_age,
        "log_annual_income": log_annual_income,
        "log_loan_amount": log_loan_amount,
        "log_total_revolving_balance": log_total_revolving_balance,
        "log_interest_to_income": log_interest_to_income,
        "log_num_hard_inquiries_6mo": log_num_hard_inquiries_6mo,
        "log_num_delinquencies_2yr": log_num_delinquencies_2yr,
    }])


    pd_score = model.predict_proba(input_df)[0][1]

    res1, res2 = st.columns(2, border=True)

    with res1:

        st.markdown("#### Probability of Default")
        st.metric("PD Score", f"{pd_score:.2%}")
        st.progress(pd_score, text=f"PD: {pd_score:.2%}")

    with res2:

        st.markdown("#### Basel III Internal Risk Grade")

        if pd_score < 0.02:
            st.success("Grade A — Prime Credit Quality")

        elif pd_score < 0.05:
            st.info("Grade B — Acceptable Risk")

        elif pd_score < 0.15:
            st.warning("Grade C — Subprime Risk")

        elif pd_score < 0.30:
            st.warning("Grade D — High Default Risk")

        else:
            st.error("Grade E — Default / Non-Performing Exposure")

        st.markdown("#### Expected Credit Loss (ECL)")

        #assumptions
        lgd = 0.45
        ead = loan_amount

        # ECL calculation
        ecl = pd_score * lgd * ead

        st.metric(
            "Expected Credit Loss",
            f"R{ecl:,.2f}"
        )

        st.caption(
            f"Calculated using PD={pd_score:.2%}, "
            f"LGD={lgd:.0%}, "
            f"EAD=R{ead:,.0f}"
        )
