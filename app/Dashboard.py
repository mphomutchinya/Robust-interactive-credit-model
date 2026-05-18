import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import load
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Business Dashboard", layout="wide")

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))

model_path = os.path.join(project_root, "models", "logistic_regression_v2.pkl")
model = load(model_path)

df_test = pd.read_csv("C:/Users/mutch_lf652j0/Credit Score Interactive Model/data/processed/engineered_test.csv")

X_test = df_test.drop(columns = ["default_flag"])
y_test = df_test["default_flag"]
y_pred_proba = model.predict_proba(X_test)[:, 1]

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

/* Popover container */
div[data-testid="stPopover"],
div[class*="menuList"],
div[class*="menu"] {
    background-color: #111827 !important;
    color: #ffffff !important;
}

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

div[data-testid="stPopover"],
div[class*="menuList"],
div[class*="menu"] {
    background-color: #111827 !important;
    color: #ffffff !important;
}
/* menu*/
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
        st.page_link("Dashboard.py",             label="Business Dashboard")
        st.page_link("pages/EDA.py",             label="EDA Tool")
        st.page_link("pages/PD_NewClient.py",    label="Probability of Default New Client")
        st.page_link("pages/Model.py",           label="Model Evaluation")
        st.divider()
        st.markdown("**Model Info**")
        st.metric("AUC", "0.7867", delta="+0.1867")
        st.metric("Gini", "0.5733")
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Train", "84,274")
        col2.metric("Test",  "36,084")

render_sidebar()

st.title("Business Dashboard")
st.markdown("Simulate lending policy decisions and explore the financial impact of model thresholds.")
st.divider()

with st.sidebar:

    st.divider()
    st.markdown("**Dashboard Controls**")
    threshold  = st.slider("Approval Threshold", 0.1, 0.9, 0.3, 0.01)
    lgd        = st.slider("LGD Assumption (%)", 20, 80, 45) / 100
    avg_loan   = st.number_input("Average Loan Amount (R)", min_value=1000, value=12000)
    margin     = st.number_input("Interest Margin (%)", min_value=1.0, value=8.0) / 100
    fli_option = st.selectbox("FLI Scenario", ["Optimistic", "Base", "Pessimistic", "Custom"])
    if fli_option == "Custom":
        fli_mult = st.number_input("Custom PD Multiplier", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
    else:
        fli_mult = {"Optimistic": 0.85, "Base": 1.00, "Pessimistic": 1.20}[fli_option]

tab1, tab2, tab3 = st.tabs([
    "Volume vs Risk",
    "Precision vs Recall",
    "ECL Calculator"
])

y_pred        = (y_pred_proba >= threshold).astype(int)
approved_mask = y_pred == 0
declined_mask = y_pred == 1
 
n_approved        = approved_mask.sum()
n_declined        = declined_mask.sum()
n_total = n_approved + n_declined
approval_rate     = n_approved / n_total
portfolio_default = y_test[approved_mask].mean() if n_approved > 0 else 0
 
y_pred_binary = (y_pred_proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
precision_val  = tp / (tp + fp + 1e-9)
recall_val     = tp / (tp + fn + 1e-9)

with tab1:
    st.markdown("### Volume vs Risk Trade-off")
    st.markdown("How many customers do we approve and at what portfolio risk?")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Approval Rate", f"{approval_rate:.1%}")
    k2.metric("Approved", f"{n_approved:,}")
    k3.metric("Declined", f"{n_declined:,}")
    k4.metric("Portfolio Default Rate", f"{portfolio_default:.2%}")
    # Threshold Sweep & Metrics 
    thresholds      = np.linspace(0.1, 0.9, 80)
    approval_rates  = []
    default_rates   = []

    for t in thresholds:
        pred   = (y_pred_proba >= t).astype(int)
        app    = (pred == 0)
        approval_rates.append(app.mean())
        default_rates.append(y_test[app].mean() if app.sum() > 0 else 0)

    approval_rates = np.array(approval_rates)
    default_rates  = np.array(default_rates)

    # Calculate the optimal Risk-Reward balance point
    d_approval = np.diff(approval_rates)
    d_default  = np.diff(default_rates)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(d_approval) > 0.001, d_default / np.abs(d_approval), 0)
    sweet_idx  = int(np.argmax(ratio)) + 1
    sweet_thresh = thresholds[sweet_idx]

    # Business Metrics for KPIs 
    # Current Baseline Metrics
    current_idx = (np.abs(thresholds - threshold)).argmin()
    curr_app_rate = approval_rates[current_idx] * 100
    curr_def_rate = default_rates[current_idx] * 100

    # Optimized Sweet Spot Metrics
    opt_app_rate = approval_rates[sweet_idx] * 100
    opt_def_rate = default_rates[sweet_idx] * 100

    # Business Deltas
    growth_delta = opt_app_rate - curr_app_rate
    risk_delta = opt_def_rate - curr_def_rate

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Optimized Approval Rate", 
            value=f"{opt_app_rate:.1f}%", 
            delta=f"{growth_delta:+.1f}% vs Current",
            delta_color="normal" if growth_delta >= 0 else "inverse"
        )

    with col2:
        st.metric(
            label="Optimized Portfolio Default Rate", 
            value=f"{opt_def_rate:.1f}%", 
            delta=f"{risk_delta:+.1f}% vs Current",
            delta_color="inverse" if risk_delta <= 0 else "normal"
        )

    with col3:
        # A calculated ratio demonstrating risk efficiency
        efficiency_ratio = opt_app_rate / (opt_def_rate if opt_def_rate > 0 else 1)
        curr_efficiency = curr_app_rate / (curr_def_rate if curr_def_rate > 0 else 1)
        eff_delta = ((efficiency_ratio - curr_efficiency) / curr_efficiency) * 100
        
        st.metric(
            label="Risk Efficiency Index", 
            value=f"{efficiency_ratio:.2f}x", 
            delta=f"{eff_delta:+.1f}% Capital Efficiency"
        )

    st.write("") # Spacer

    # Dual-Axis Chart Generation
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    # Plot Business curves 
    ax1.plot(thresholds, approval_rates * 100, color="#F58220", linewidth=2.5, label="Customer Approval Rate %")
    ax2.plot(thresholds, default_rates  * 100, color="#e05252", linewidth=2.5, label="Portfolio Default Rate %", linestyle="--")

    # Policy benchmarks
    ax1.axvline(threshold,    color="#9CA3AF", linestyle=":",  linewidth=1.5, label=f"Current Policy ({threshold:.2f})")
    ax1.axvline(sweet_thresh, color="#00d084", linestyle="--", linewidth=2,   label=f"Optimized Policy ({sweet_thresh:.2f})")

    # Labels & Business Terminology
    ax1.set_xlabel("Model Risk Tolerance Level (Threshold Score)")
    ax1.set_ylabel("Customer Approval Rate %",        color="#F58220", fontweight="bold")
    ax2.set_ylabel("Portfolio Default Rate %", color="#e05252", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#F58220")
    ax2.tick_params(axis="y", labelcolor="#e05252")

    # Combine legends into a clean block
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
            facecolor="#111827", labelcolor="white", edgecolor="#374151", fontsize=9, loc="upper left")

    # Annotation pointing out the strategic pivot point
    ax1.annotate(f"Optimized Policy\n(Threshold: {sweet_thresh:.2f})",
                xy=(sweet_thresh, approval_rates[sweet_idx] * 100),
                xytext=(sweet_thresh + 0.04, approval_rates[sweet_idx] * 100 + 7),
                color="#00d084", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#00d084", lw=1.5))

    # Premium Dark Theme Styling 
    fig.patch.set_facecolor("#0B0F14")
    ax1.set_facecolor("#111827")
    ax1.tick_params(colors="white")
    ax2.tick_params(colors="white")
    ax1.xaxis.label.set_color("white")

    for spine in ax1.spines.values(): spine.set_edgecolor("#374151")
    for spine in ax2.spines.values(): spine.set_edgecolor("#374151")

    
    st.pyplot(fig)
    st.divider()

    st.info(
    f"**Executive Summary:** By shifting the credit policy cutoff from the current baseline "
    f"({threshold:.2f}) to the mathematically optimized **Optimized Profitability Point ({sweet_thresh:.2f})**, "
    f"the portfolio can achieve a **{growth_delta:+.1f}% change in customer approvals** "
    f"while shifting portfolio default risk by **{risk_delta:+.1f}%**.")

     # Insight box
    curr_idx    = np.argmin(np.abs(thresholds - threshold))
    lower_idx   = max(curr_idx - 8, 0)
    delta_apps  = int((approval_rates[curr_idx] - approval_rates[lower_idx]) * n_total)
    delta_risk  = (default_rates[curr_idx] - default_rates[lower_idx]) * 100
 
    st.info(
        f"**At threshold {threshold:.2f}**, you approve **{approval_rate:.1%}** of applicants "
        f"with a portfolio default rate of **{portfolio_default:.2%}**.\n\n"
        f"Lowering the threshold by 0.10 adds approximately **{abs(delta_apps):,} approvals** "
        f"but increases portfolio risk by **{abs(delta_risk):.2f}%**.\n\n"
        f"The optimal sweet spot is around **{sweet_thresh:.2f}** — beyond this point, "
        f"default risk accelerates faster than approval volume grows."
    )

with tab2:
    st.markdown("### Precision vs Recall — Business Interpretation")
    left, right = st.columns(2)
    
    with left:
        st.markdown("#### Precision")
        st.markdown("Of all **predicted defaulters**, how many actually defaulted?")
        st.metric("Risk Precision", f"{precision_val:.2%}")
        st.caption(f"We mistakenly declined {int(fp):,} good customers (False Positives).")

    with right:
        st.markdown("#### Recall")
        st.markdown("Of all **actual defaulters**, how many did our model catch?")
        st.metric("Defaulters Caught", f"{int(tp):,}")
        st.metric("Model Recall", f"{recall_val:.2%}")
        st.caption(f"We missed {int(fn):,} defaulters who were approved (False Negatives).")

    st.divider()


    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    curr_prec = precision_score(y_test, y_pred_binary, zero_division=0)
    curr_rec  = recall_score(y_test, y_pred_binary, zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rec_curve, prec_curve, color="#F58220", linewidth=2.5, label="Precision-Recall Curve")
    ax.scatter(curr_rec, curr_prec, color="#00d084", s=120, zorder=5, 
               label=f"Active Decision Boundary ({threshold:.2f})")

    ax.set_xlabel("Recall (Sensitivity to Default)")
    ax.set_ylabel("Precision (Accuracy of Risk Flags)")
    ax.legend(facecolor="#111827", labelcolor="white", edgecolor="#374151", fontsize=9)
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values(): spine.set_edgecolor("#374151")

    st.pyplot(fig)
    st.divider()

    
    cost_fn = fn * lgd * avg_loan
    cost_fp = fp * avg_loan * margin

    error_df = pd.DataFrame({
        "Error Metric":       ["False Negative (Type II error)", "False Positive (Type I error)"],
        "Operational Meaning": ["Approved an applicant who defaulted", "Declined a good applicant"],
        "Account Count":       [int(fn), int(fp)],
        "Unit Financial Impact": [f"R{lgd * avg_loan:,.0f} (Credit Loss)", f"R{avg_loan * margin:,.0f} (Lost Margin)"],
        "Total Portfolio Impact": [f"R{cost_fn:,.0f}", f"R{cost_fp:,.0f}"]
    })
    st.dataframe(error_df, use_container_width=True, hide_index=True)
    
    st.divider()

    if cost_fn > cost_fp:
        st.warning(
            f"**Policy Alert at Threshold {threshold:.2f}:** Credit write-offs from bad approvals "
            f"**({error_df.loc[0, 'Total Portfolio Impact']})** exceed missed opportunity costs. "
            f"**Recommendation:** Raise the threshold cutoff to tighten credit policy."
        )
    else:
        st.success(
            f"💡 **Policy Optimization Window at Threshold {threshold:.2f}:** Revenue lost from turning away viable borrowers "
            f"**({error_df.loc[1, 'Total Portfolio Impact']})** exceeds actual credit write-offs. "
            f"**Recommendation:** Lower the threshold cutoff to safely capture more market share."
        )

with tab3:

    st.markdown("## IFRS 9 Expected Credit Loss (ECL) Simulator")
    st.markdown(
        "Evaluate your portfolio's forward-looking credit impairments under different loss and exposure scenarios."
    )

    approved_pds = y_pred_proba[approved_mask]
    total_approved = int(len(approved_pds))

    if total_approved == 0:
        st.warning(" No applicants are currently approved under your active credit policy threshold. Adjust your policy to simulate ECL.")

    else:

        st.info(
            "### Accounting Framework: Regulation Standard\n"
            "$$\\text{ECL} = \\text{PD (Probability of Default)} \\times \\text{LGD (Loss Given Default)} \\times \\text{EAD (Exposure at Default)}$$"
        )
        st.divider()

        # Metrics derived from inputs (sidebar values: lgd, avg_loan)
        portfolio_pd = np.mean(approved_pds) if total_approved > 0 else 0
        total_ead = total_approved * avg_loan
        total_ecl_val = portfolio_pd * lgd * total_ead

        stage1_mask = approved_pds < 0.10
        stage2_mask = (approved_pds >= 0.10) & (approved_pds <= 0.30)
        stage3_mask = approved_pds > 0.30

        counts = [np.sum(stage1_mask), np.sum(stage2_mask), np.sum(stage3_mask)]
        pds_by_stage = [
            np.mean(approved_pds[stage1_mask]) if counts[0] > 0 else 0,
            np.mean(approved_pds[stage2_mask]) if counts[1] > 0 else 0,
            np.mean(approved_pds[stage3_mask]) if counts[2] > 0 else 0
        ]

        # Calculate individual ECL per stage 
        ecls_by_stage = [
            counts[0] * pds_by_stage[0] * lgd * avg_loan,
            counts[1] * pds_by_stage[1] * lgd * avg_loan,
            counts[2] * pds_by_stage[2] * lgd * avg_loan
        ]

        st.write("### Portfolio Impairment Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(label="Weighted Portfolio PD", value=f"{portfolio_pd:.2%}", help="Average default risk of approved clients.")
        c2.metric(label="Loss Given Default (LGD)", value=f"{lgd:.1%}", help="Percentage of exposure lost if a client defaults.")
        c3.metric(label="Total Portfolio EAD", value=f"R{total_ead:,.0f}", help="Total capital exposed across all approved accounts.")
        c4.metric(label="Total Impairment Provision (ECL)", value=f"R{total_ecl_val:,.0f}", help="Total required capital buffer for credit losses.")
        
        st.divider()

        left_layout, right_layout = st.columns([3, 2])

        with left_layout:
            st.write("### IFRS 9 Staging Ledger")
            
            # DF
            staging_df = pd.DataFrame({
                "Stage": ["Stage 1 (Performing)", "Stage 2 (Underperforming)", "Stage 3 (Non-Performing)"],
                "Criteria": ["PD < 10%", "PD 10% - 30%", "PD > 30%"],
                "Count": [int(c) for c in counts],
                "% Portfolio": [f"{(c / total_approved):.1%}" if total_approved > 0 else "0.0%" for c in counts],
                "Weighted PD": [f"{p:.2%}" for p in pds_by_stage],
                "ECL Provision": [f"R{e:,.0f}" for e in ecls_by_stage]
            })
            
            st.dataframe(staging_df, use_container_width=True, hide_index=True)

        with right_layout:
            st.write("### Visual Provisions Distribution")
            
            # Build Stacked Bar Chart for ECL breakdown by stage
            fig, ax = plt.subplots(figsize=(5, 4.3))
            
            # Risk asset colors matching dark mode aesthetic
            stage_colors = ["#00d084", "#F58220", "#FF4B4B"] # Green, Orange, Red
            stages = ["Stage 1", "Stage 2", "Stage 3"]
            
            # Convert values to Thousands for cleaner chart axes label readability
            ecl_thousands = [e / 1000 for e in ecls_by_stage]
            
            bars = ax.bar(stages, ecl_thousands, color=stage_colors, edgecolor="#374151", width=0.6)
            
            # Label each bar with exact values to limit visual strain
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'R{height:,.0f}k',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', color='white', fontsize=9)

            ax.set_ylabel("Impairment Provisions (R Thousands)", color="white")
            fig.patch.set_facecolor("#0B0F14")
            ax.set_facecolor("#111827")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#374151")
                
            st.pyplot(fig)

