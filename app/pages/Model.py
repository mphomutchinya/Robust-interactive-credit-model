import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, os
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay)
from scipy.stats import ks_2samp
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
        st.metric("AUC", "0.7867", delta="+0.1867")
        st.metric("Gini", "0.5733")
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Train", "84 274")
        col2.metric("Test",  "36 084")

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance", "Threshold analysis", "Feature effects", "Risk Separation", "Benchmark vs Model"])

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
    
    st.markdown("### Feature Coefficients")
    st.markdown("Coefficients represent the weight each feature has in the model. "
                "Positive coefficients increase the probability of default, negative decrease it.")

    # Extract coefficients
    coef_df = pd.DataFrame({
        "Feature":     X_test.columns,
        "Coefficient": model.coef_[0]
    })

    # Sort by absolute value
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=True)

    # Colour: red for positive, green for negative
    colours = ["#e05252" if c > 0 else "#00d084" for c in coef_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colours)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Logistic Regression Coefficients")

    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    st.pyplot(fig)

    st.divider()

    # Table
    st.markdown("#### Coefficient Table")
    coef_table = coef_df[["Feature", "Coefficient"]].sort_values("Coefficient", ascending=False)
    coef_table["Direction"] = coef_table["Coefficient"].apply(
        lambda x: "↑ Increases Risk" if x > 0 else "↓ Decreases Risk"
    )
    coef_table["Coefficient"] = coef_table["Coefficient"].round(4)
    st.dataframe(coef_table, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### Score Distribution")
    st.markdown("Shows how well the model separates defaulters from non-defaulters. "
                "A good model produces clearly separated distributions.")

    # Split scores by actual class
    scores_default     = y_pred_proba[y_test == 1]
    scores_nondefault  = y_pred_proba[y_test == 0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(scores_nondefault, bins=50, color="#00d084", alpha=0.5, 
            label="Non-Default", density=True)
    ax.hist(scores_default,    bins=50, color="#e05252", alpha=0.5, 
            label="Default",     density=True)

    ax.axvline(0.3, color="white", linestyle="--", linewidth=1.2, label="Threshold = 0.30")
    ax.set_xlabel("Predicted Probability of Default")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution — Default vs Non-Default")
    ax.legend(facecolor="#111827", labelcolor="white", edgecolor="#374151")

    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    st.pyplot(fig)

    st.divider()

    # KS Statistic 
    st.markdown("#### KS Statistic")

    ks_stat, ks_pvalue = ks_2samp(scores_default, scores_nondefault)

    k1, k2, k3 = st.columns(3)
    k1.metric("KS Statistic",  f"{ks_stat:.4f}")
    k2.metric("KS p-value",    f"{ks_pvalue:.2e}")
    k3.metric("Interpretation", 
              "Strong" if ks_stat > 0.3 else "Moderate" if ks_stat > 0.2 else "Weak")

    st.divider()

    #  Context 
    st.markdown("#### What does this mean?")
    if ks_stat > 0.3:
        st.success(
            f"KS = {ks_stat:.4f} — The model strongly separates defaulters from non-defaulters. "
            f"The two score distributions are clearly distinct, indicating strong discriminatory power."
        )
    elif ks_stat > 0.2:
        st.warning(
            f"KS = {ks_stat:.4f} — Moderate separation between classes. "
            f"The model distinguishes risk reasonably well but there is overlap between groups."
        )
    else:
        st.error(
            f"KS = {ks_stat:.4f} — Weak separation. "
            f"The model struggles to distinguish defaulters from non-defaulters."
        )

with tab5:

    st.markdown("### Baseline Comparison")
    st.markdown("Comparing our logistic regression scorecard against the legacy baseline model.")

    # Comparison Table 
    st.markdown("#### Performance Metrics")

    comparison_df = pd.DataFrame({
        "Metric":      ["AUC", "Gini", "Accuracy", "Precision (Default)", "Recall (Default)", "F1 (Default)"],
        "Baseline":    [0.68,   0.36,   0.79,        0.35,                  0.30,               0.32],
        "Our Model":   [0.7867 , 0.5733, 0.83,        0.43,                  0.40,               0.42],
        "Improvement": ["+0.1067", "+0.2133", "+0.04", "+0.08",            "+0.10",            "+0.10]"]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Visual bar comparison 
    st.markdown("#### AUC Comparison")

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(
        ["Baseline", "Our Model", "LightGBM Ceiling"],
        [0.68, 0.7813, 0.82],
        color=["#374151", "#F58220", "#00d084"]
    )
    ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("AUC Score")
    ax.axvline(0.68,   color="#374151", linestyle="--", linewidth=0.8)
    ax.axvline(0.7867, color="#F58220", linestyle="--", linewidth=0.8)
    ax.axvline(0.82,   color="#00d084", linestyle="--", linewidth=0.8)

    for bar, val in zip(bars, [0.68, 0.7813, 0.82]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="white", fontsize=10)

    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    st.pyplot(fig)

    st.divider()

    # Feature engineering contribution 
    st.markdown("#### What drove the improvement?")

    contributions = {
    "Log Transforms": "Compressing skewed features like annual income and loan amount reduced the influence of extreme outliers.",
    "Interaction Features": "rate_to_age and rate_dti_burden captured combined risk signals that neither interest rate nor age/DTI could express independently.",
    "Missing Value Flags": "Treating missingness as a signal (missing_annual_income, missing_months_since_last_delinquency) allowed the model to learn that unknown data itself correlates with default risk.",
    "Hard Inquiries Binning": "Binning hard_inquiries × delinquencies into risk tiers (0–3) reduced noise and made the interaction term more stable.",
    "Multicollinearity Removal": "Dropping highly correlated features (dti_ratio, months_since_oldest_account) reduced redundancy and improved coefficient stability.",
    "Delinquency Recency Bucketing": "Converting months_since_last_delinquency into ordinal risk buckets (0=never delinquent, 3=most recent) produced a clean monotonic relationship with default rate — from 8% to 29%.",
    "Sentinel Value Imputation": "Filling missing months_since_last_delinquency with 900 preserved the distinction between never-delinquent and historically-delinquent applicants, avoiding information loss from standard imputation.",
    "Credit History to Age Ratio": "Combining months_since_oldest_account and age into a single ratio resolved the 0.95 multicollinearity between them while creating a more meaningful feature — how long relative to their life has the applicant had credit.",
    "Outlier Capping": "IQR-based capping on heavily skewed columns like annual_income and total_revolving_balance prevented extreme values from distorting the logistic regression decision boundary.",
    "Low Predictive Feature Removal": "Dropping categorical variables with IV below 0.02 (region, application_dow, email_domain_type) and uninformative columns (phone_verified, months_at_current_address, branch_code_id) reduced noise and simplified the model."
}

    for title, desc in contributions.items():
        with st.expander(f"**{title}**"):
            st.write(desc)



 