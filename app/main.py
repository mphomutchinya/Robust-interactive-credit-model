# 0 is no default(low risk), 1 is default(high risk)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(page_title="Credit Default Interactive Dashboard", layout="wide")

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

.stButton > button:hover {
    background-color: #ff9a3c !important;
    color: white !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #F58220;
    padding: 15px;
    border-radius: 12px;
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
        st.page_link("main.py", label="Overview")
        st.page_link("pages/EDA.py", label="EDA Tool")
        st.page_link("pages/PD_NewClient.py", label= "Probability of Default New Client")
        st.page_link("pages/Model.py", label="Model Evaluation")
        st.page_link("pages/Dashboard.py", label="Business Dashboard")
        st.divider()
        st.markdown("**Model Info**")
        st.metric("AUC", "0.7813", delta="+0.1013")
        st.metric("Gini", "0.5627")
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Train", "84,274")
        col2.metric("Test", "36,084")

render_sidebar()

st.title("Credit Default Interactive Dashboard")
