import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

df_cleaned = pd.read_csv("C:/Users/mutch_lf652j0/Credit Score Interactive Model/data/processed/cleaned_train.csv")
df_raw = pd.read_csv("C:/Users/mutch_lf652j0/Credit Score Interactive Model/data/raw/loan_book.csv")

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
        col1.metric("Train", "84 274")
        col2.metric("Test",  "36 084")

render_sidebar()


st.title("Exploratory Data Analysis Tool")
st.markdown("Explore the relationship between features and their effects on defaulting")

tabs1, tabs2, tabs3, tabs4 = st.tabs(["Univariate", "Bivariate", "Default Rate", "Data Quality(Raw)"])

features = ['age',
    'annual_income',
    'employment_length_years',
    'home_ownership',
    'region',
    'num_open_accounts',
    'num_delinquencies_2yr',
    'total_revolving_balance',
    'credit_utilisation_pct',
    'months_since_oldest_account',
    'num_hard_inquiries_6mo',
    'loan_amount',
    'interest_rate',
    'loan_purpose',
    'dti_ratio',
    'months_since_last_delinquency',
    'pct_accounts_current',
    'application_dow',
    'email_domain_type',
    'default_flag',
    'missing_annual_income',
    'missing_months_since_last_delinquency']

with tabs1:

    left_col, right_col = st.columns(2, border = True)

    with left_col:

        selected_feature = st.selectbox("Select a feature", features)

        plot_type = st.selectbox("Choose plot type", ["Histogram", "KDE plot", "Box plot", "Default Overlay"])

    with right_col:

        fig, ax = plt.subplots(figsize = (6,4))

        #Histogram
        if plot_type == "Histogram":
            sns.histplot(df_cleaned[selected_feature], kde = True, color = "#F58220")
        #KDE
        elif plot_type == "KDE plot":
            sns.kdeplot(df_cleaned[selected_feature], color = "#F58220", fill=True)
        #Boxplot
        elif plot_type == "Box plot":
            sns.boxplot(df_cleaned[selected_feature], color = "#F58220", fill=True)
        #Default overlay
        elif plot_type == "Default Overlay":
            default_dist = df_cleaned[df_cleaned["default_flag"] == 1][selected_feature]
            nondefault_dist = df_cleaned[df_cleaned["default_flag"] == 0][selected_feature]

            if df_cleaned[selected_feature].dtype == "object" or df_cleaned[selected_feature].nunique() <= 10:
                default_rate = df_cleaned.groupby(selected_feature)["default_flag"].mean().sort_values()
                ax.barh(default_rate.index.astype(str), default_rate.values, color="#F58220", alpha=0.8)
                ax.axvline(df_cleaned["default_flag"].mean(), color="#00d084", linestyle="--", label="Overall avg")
                ax.set_xlabel("Default Rate")
                ax.legend(facecolor="#111827", labelcolor="white", edgecolor="#374151")
            else:
                sns.kdeplot(default_dist, color="#F58220", fill=True, alpha=0.3, label="Default", ax=ax)
                sns.kdeplot(nondefault_dist, color="#00d084", fill=True, alpha=0.3, label="Non-Default", ax=ax)
                ax.legend(facecolor="#111827", labelcolor="white", edgecolor="#374151")

        st.pyplot(fig)

        if plot_type == "Default Overlay":

            if df_cleaned[selected_feature].dtype == "object":
                st.markdown("#### Value Counts")
                st.dataframe(df_cleaned[selected_feature].value_counts().reset_index()
                            .rename(columns={selected_feature: "Category", "count": "Count"}))
         
            else:

                default_mean = default_dist.mean()
                nondefault_mean = nondefault_dist.mean()
                difference = default_mean - nondefault_mean

                mean1, mean2, d = st.columns(3)
                mean1.metric(f"Default Mean",  f"{default_mean:.2f}")
                mean2.metric(f"Non Default Mean", f"{nondefault_mean:.2f}")
                d.metric(f"Difference", f"{difference:.2f}")

with tabs2: 

    left, right = st.columns(2, border = True)

    with left:

        featurex = st.selectbox("Select X variable", features, key="x_feature")
        featurey = st.selectbox("Select Y variable", features, key="y_feature")


        if (pd.api.types.is_numeric_dtype(df_cleaned[featurex]) and pd.api.types.is_numeric_dtype(df_cleaned[featurey])):
    
            biplot_type = st.selectbox("Choose plot type", ["Scatter Plot", "Hexbin Plot"])

            if biplot_type in ["Scatter Plot","Hexbin Plot"]:
                show_regression = st.checkbox("Fit linear regression line")
                remove_outliers = st.checkbox("Remove outliers (99th percentile)")

        elif ((df_cleaned[featurex].dtype == "object" and df_cleaned[featurey].dtype == "float64") or df_cleaned[featurex].dtype == "float64" and df_cleaned[featurey].dtype == "object"):
    
            biplot_type = st.selectbox("Choose plot type", ["Violin Plot", "Box Plot"])
        
        elif(df_cleaned[featurex].dtype == "object" and df_cleaned[featurey].dtype == "object"):

            biplot_type = st.selectbox("Choose plot type", ["Heatmap"])


#For scatterplot, we will use a sample since the data is too large

        
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))

        if biplot_type == "Scatter Plot":
            df_cleaned_sample = df_cleaned.sample(n=2000, random_state=42)

            if remove_outliers:
                q99x = df_cleaned_sample[featurex].quantile(0.99)
                q99y = df_cleaned_sample[featurey].quantile(0.99)
                df_cleaned_sample = df_cleaned_sample[
                (df_cleaned_sample[featurex] <= q99x) &
                (df_cleaned_sample[featurey] <= q99y)
            ]

            ax.scatter(df_cleaned_sample[featurex], df_cleaned_sample[featurey],
                   color="#F58220", alpha=0.4)
            ax.set_xlabel(featurex)
            ax.set_ylabel(featurey)
            ax.set_title(f"{featurex} vs {featurey}")
        
            st.caption("Showing a random sample of 2000 records")

            if show_regression:
                X = df_cleaned_sample[[featurex]].values
                y = df_cleaned_sample[featurey].values
                model = LinearRegression()
                model.fit(X, y)
                x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_line = model.predict(x_line)
                ax.plot(x_line, y_line, color="#00d084", linewidth=2,
                    label=f"Linear fit (R²={model.score(X, y):.3f})")
                ax.legend(facecolor="#111827", labelcolor="white", edgecolor="#374151")
                

        elif biplot_type == "Hexbin Plot":
            df_cleaned_sample = df_cleaned.sample(n=2000, random_state=42)
            hb = ax.hexbin(df_cleaned_sample[featurex], df_cleaned_sample[featurey],
                gridsize=30, cmap="Oranges")
            fig.colorbar(hb, ax=ax, label="Count")
            ax.set_xlabel(featurex)
            ax.set_ylabel(featurey)
            ax.set_title(f"{featurex} vs {featurey}")

            st.caption("Showing a random sample of 2000 records")

        elif biplot_type == "Violin Plot":
            sns.violinplot(data = df_cleaned, x=featurex, y=featurey, color="#F58220", ax=ax)
            ax.set_xlabel(featurex)
            ax.set_ylabel(featurey)
            ax.set_title(f"{featurex} vs {featurey}")

        elif biplot_type == "Box Plot":
            sns.boxplot(data = df_cleaned, x = featurex, y = featurey, color = "#F58220", ax=ax)
            ax.set_xlabel(featurex)
            ax.set_ylabel(featurey)
            ax.set_title(f"{featurex} vs {featurey}")

        elif biplot_type == "Heatmap":
            contingency = pd.crosstab(df_cleaned[featurex], df_cleaned[featurey], 
                normalize="index")
            sns.heatmap(contingency, annot=True, fmt=".2f", cmap="Oranges", ax=ax)
            ax.set_title(f"{featurex} vs {featurey}")

        st.pyplot(fig)

        if biplot_type in ["Scatter Plot", "Hexbin Plot"]:
            pearson_r = df_cleaned[[featurex, featurey]].corr().iloc[0, 1]
            spearman_r = df_cleaned[[featurex, featurey]].corr(method="spearman").iloc[0, 1]
            col1, col2 = st.columns(2)
            col1.metric("Pearson Correlation", f"{pearson_r:.3f}")
            col2.metric("Spearman Correlation", f"{spearman_r:.3f}")

with tabs3:

    left_cols, right_cols = st.columns(2, border = True)

    with left_cols:

        variable = st.selectbox("Select a feature", features, key = "tab3 feature")
        n_bins = st.slider("Number of bins", min_value=3, max_value=15, value=5, key="tab3_bins")

    with right_cols:

        fig, ax = plt.subplots(figsize=(6, 4))
        if df_cleaned[variable].dtype == "object":  
            encoded = df_cleaned[variable].astype("category").cat.codes
        else:
            encoded = df_cleaned[variable]

        # Bin the feature
        df_temp = df_cleaned[["default_flag"]].copy()
        df_temp["binned"] = pd.cut(encoded, bins=n_bins)
        default_rate = df_temp.groupby("binned", observed=True)["default_flag"].mean()
        counts = df_temp.groupby("binned", observed=True)["default_flag"].count()

        bars = ax.bar(range(len(default_rate)), default_rate.values, color="#F58220", alpha=0.8)
        ax.axhline(df_cleaned["default_flag"].mean(), color="#00d084", 
            linestyle="--", linewidth=1.5, label="Overall avg")
        ax.set_xticks(range(len(default_rate)))
        ax.set_xticklabels([str(b) for b in default_rate.index], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Default Rate")
        ax.set_title(f"Default Rate by {variable} bins")
        ax.legend(facecolor="#111827", labelcolor="white", edgecolor="#374151")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

        st.pyplot(fig)

        # Metrics
        highest_bin = default_rate.idxmax()
        lowest_bin = default_rate.idxmin()

        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Default Rate", f"{df_cleaned['default_flag'].mean():.2%}")
        m2.metric("Highest Risk Bin",     f"{default_rate.max():.2%}")
        m3.metric("Lowest Risk Bin",      f"{default_rate.min():.2%}")

        # Table
        st.markdown("#### Bin Summary")
        summary = pd.DataFrame({
            "Bin":          default_rate.index.astype(str),
            "Default Rate": default_rate.values,
            "Count":        counts.values
        })
        summary["Default Rate"] = summary["Default Rate"].apply(lambda x: f"{x:.2%}")
        st.dataframe(summary, use_container_width=True)
        
with tabs4:

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Rows", f"{len(df_raw):,}")
    k2.metric("Total Columns", f"{df_raw.shape[1]:,}")
    k3.metric("Duplicate Rows",   f"{df_raw.duplicated().sum():,}")
    k4.metric("Columns with Nulls", f"{df_raw.isnull().any().sum()}")

    st.divider()

    left_col, right_col = st.columns(2, border=True)

    with left_col:
        st.markdown("#### Missing Values by Column")

        missing_data = df_raw.isnull().sum()
        missing_percentage = (missing_data[missing_data>0] / df_raw.shape[0]) * 100
        missing_percentage.sort_values(ascending = True, inplace = True)

        #plot chart
        fig, ax = plt.subplots(figsize = (8,6))
        ax.barh(missing_percentage.index, missing_percentage, color = "#F58220")

        # Annotate the values and indexes
        for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
            ax.text(value+0.5, i, f"{value:.2f}%", ha='left', va='center', fontweight='bold', fontsize=18, color='black')

        ax.set_xlim([0, 50])

        plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
        plt.xlabel('Percentages (%)', fontsize=16)
        st.pyplot(fig)

    with right_col:
        
        st.markdown("#### Column Summary")
        summary = pd.DataFrame({
            "Column":   df_raw.columns,
            "Type":     df_raw.dtypes.astype(str).values,
            "Nulls":    df_raw.isnull().sum().values,
            "Unique":   df_raw.nunique().values,
            "Mean":     df_raw.mean(numeric_only=True).reindex(df_raw.columns).round(2).values
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)














