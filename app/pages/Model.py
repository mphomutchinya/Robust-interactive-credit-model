import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, os
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "logistic_regression_v2.pkl")
model = load(model_path)

df_test = pd.read_csv("C:/Users/mutch_lf652j0/Credit Score Interactive Model/data/processed/engineered_test.csv")

X_test = df_test.drop(columns = ["default_flag"])
y_test = df_test["default_flag"]
y_pred_proba = model.predict_proba(X_test)[:, 1]

st.set_page_config(page_title="Model Evaluation", layout="wide")

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
        st.page_link("main.py",            label="Overview")
        st.page_link("pages/EDA.py",       label="EDA Tool")
        st.page_link("pages/PD_NewClient.py", label= "Probability of Default New Client")
        st.page_link("pages/Model.py",     label="Model Evaluation")
        st.page_link("pages/Dashboard.py", label="Business Dashboard")
        st.divider()
        st.markdown("**Model Info**")
        st.metric("AUC", "0.7813", delta="+0.1013")
        st.metric("Gini", "0.5627")
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Train", "84,274")
        col2.metric("Test",  "36,084")

render_sidebar()

st.title("Model Evaluation")

st.markdown("Evaluation of the model and the performance")

auc_score = roc_auc_score(y_test, y_pred_proba)
gini = 2 * auc_score - 1
auc_lift = auc_score - 0.68

e1,e2,e3,e4 = st.columns(4)

e1.metric(label="AUC Score", value=f"{auc_score:.4f}")
e2.metric(label="Gini Coefficient", value=f"{gini:.4f}")
e3.metric(label="Benchmark AUC", value="0.6800")
e4.metric(label="AUC Lift", value=f"{auc_lift:.4f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Performance", "Threshold analysis", "Feature effects", "Risk Separation", "Model equation", "Benchmarking"])

with tab1:

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Logistic Regression')
    ax.legend(loc='lower right')
    fig.tight_layout()

    col1, col2, = st.columns([1, 2])

    with col1:
    
        st.pyplot(fig, use_container_width=True)

    with col2:

        score1, score2 = st.columns(2)

        ks = np.max(tpr - fpr)

        score1.metric(label="AUC Score", value=f"{auc_score:.4f}")
        score2.metric(label="KS Statistic", value=f"{ks:.4f}")

        st.divider()

        st.markdown("AUC measures how well the model can distinguish between different classes, a score of 0.7867 provides significant value in separating the classes.")
        st.markdown("KS statistic measures the separation power of the model, values above 40 % can indicate strong predictive power.")

with tab2:

    colss1, colss2 = st.columns(2, border = True)

    with colss1:
        
        thresh = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        st.write("Determines the probability threshold at which a client is classified as default")

    with colss2:

        y_pred_thresh = (y_pred_proba >= thresh).astype(int)

        cm = confusion_matrix(y_test, y_pred_thresh)
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown("#### Confusion Matrix")
        cm_df = pd.DataFrame(
            [[tp, fp], [fn, tn]],
            index=["Actual Default", "Actual Non-Default"],
            columns=["Predicted Default", "Predicted Non-Default"]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.divider()

        # Precision, Recall, F1
        st.markdown("#### Classification Metrics")
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * (precision * recall) / (precision + recall + 1e-9)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{precision:.2%}")
        m2.metric("Recall",    f"{recall:.2%}")
        m3.metric("F1 Score",  f"{f1:.2%}")

        st.divider()

        
        st.markdown("#### Business Interpretation")
        if precision > recall:
            st.info(
                f"**Conservative policy** — At threshold {thresh:.2f}, the model is more precise "
                f"but misses more defaulters. Fewer false alarms but higher credit risk exposure. "
                f"Best when the cost of approving a bad loan is high."
            )
        else:
            st.warning(
                f"**Aggressive policy** — At threshold {thresh:.2f}, the model catches more "
                f"defaulters but flags more good clients incorrectly. Higher rejection rate "
                f"but lower credit risk. Best when the cost of a bad loan is very high."
            )
            
with tab3:

    

