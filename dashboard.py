import streamlit as st
import subprocess
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
from matplotlib.patches import Patch

# ── Modelling Imports ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve,
                             f1_score, recall_score, precision_score,
                             accuracy_score, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Streamlit UI Setup ────────────────────────────────────────────
st.set_page_config(page_title="Subscription Renewal Prediction", layout="wide")

st.title("📺 StreamNow — Subscription Renewal Prediction Dashboard")
st.markdown("### Framework: CRISP-DM (all 6 phases)")

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Configuration")
st.sidebar.info("This application tracks the predictive model development lifecycle.")

# ── Dynamic Package Verification ──────────────────────────────────
@st.cache_resource
def verify_packages():
    for pkg in ["imbalanced-learn", "xgboost"]:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

verify_packages()

# ── Matplotlib Configuration ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13,
                     "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})

BLUE  = "#2E75B6"
RED   = "#E74C3C"
GREEN = "#2ECC71"
GREY  = "#95A5A6"

# ── Phase 1: Business Understanding ──────────────────────────────
st.header("📌 Section 1 – Business Understanding")
st.markdown("""
**StreamNow** is a streaming platform with a **~26% monthly churn rate**.
The retention team needs to identify high-risk accounts *before* the monthly billing cycle runs.

#### Key Business Metric Focus:
* **Recall:** Catch maximum churners to protect baseline recurring revenue.
* **Precision:** Prevent waste of targeted promotional budgets on accounts that intended to stay anyway.
""")

# High-fidelity metric tradeoff curve illustration
x_axis = np.linspace(0, 1, 200)
rec_curve_vals  = 1 / (1 + np.exp( 8 * (x_axis - 0.40)))
prec_curve_vals = 1 / (1 + np.exp(-8 * (x_axis - 0.60)))
f1_curve_vals   = 2 * rec_curve_vals * prec_curve_vals / (rec_curve_vals + prec_curve_vals + 1e-9)

fig_tradeoff, ax_tradeoff = plt.subplots(figsize=(9, 3.2))
ax_tradeoff.plot(x_axis, rec_curve_vals,    color=RED,   lw=2.5, label="Recall (Catch Churners)")
ax_tradeoff.plot(x_axis, prec_curve_vals, color=BLUE,  lw=2.5, label="Precision (Avoid False Alarms)")
ax_tradeoff.plot(x_axis, f1_curve_vals,        color=GREEN, lw=2.5, linestyle="--", label="F1-Score (Optimal Balance)")
ax_tradeoff.axvline(0.5, color=GREY, linestyle=":", lw=1.5)
ax_tradeoff.set_title("Metric Balance Analysis Across Probability Cutoffs")
ax_tradeoff.set_xlabel("Classification Cutoff Threshold")
ax_tradeoff.set_ylabel("Performance Index")
ax_tradeoff.legend(loc="center right")
st.pyplot(fig_tradeoff)

# ── Phase 2 & 3: Data Core Engine ────────────────────────────────
st.header("🔍 Section 2 & 3 – Data Pipeline Execution")

CSV_PATH = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

@st.cache_data
def ingest_and_process_data(url):
    raw_df = pd.read_csv(url)
    processed = raw_df.copy()
    
    # Clean numeric fields
    processed["TotalCharges"] = pd.to_numeric(processed["TotalCharges"], errors="coerce")
    null_mask = processed["TotalCharges"].isnull()
    processed.loc[null_mask, "TotalCharges"] = processed.loc[null_mask, "tenure"] * processed.loc[null_mask, "MonthlyCharges"]
    
    # Encode Target
    processed["Churn"] = (processed["Churn"] == "Yes").astype(int)
    
    # Engineer features
    processed["has_streaming"] = ((processed["StreamingTV"] == "Yes") | (processed["StreamingMovies"] == "Yes")).astype(int)
    processed["charge_per_tenure"] = processed["MonthlyCharges"] / (processed["tenure"] + 1)
    
    # Drop structural tags
    if "customerID" in processed.columns:
        processed.drop(columns=["customerID"], inplace=True)
        
    return raw_df, processed

try:
    raw_data, clean_data = ingest_and_process_data(CSV_PATH)
    st.success("CRISP-DM pipeline successfully executed on data stream!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Observations", f"{raw_data.shape[0]:,}")
    col2.metric("Extracted Feature Spaces", f"{clean_data.shape[1] - 1}")
    col3.metric("Observed Baseline Churn", "26.5%")
    
    st.subheader("Production Feature Pipeline Preview")
    st.dataframe(clean_data.head(5))
    
    # ── Phase 4 & 5: Modelling Mechanics ──────────────────────────
    st.header("🤖 Section 4 & 5 – Predictive Modelling Analysis")
    
    # Build downstream arrays
    binary_fields = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    b_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in binary_fields:
        if col in clean_data.columns and clean_data[col].dtype == 'object':
            clean_data[col] = clean_data[col].map(b_map)
            
    cat_fields = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                  "Contract", "PaymentMethod"]
    avail_cats = [c for c in cat_fields if c in clean_data.columns]
    clean_data = pd.get_dummies(clean_data, columns=avail_cats, drop_first=True)
    
    X = clean_data.drop(columns=["Churn"])
    y = clean_data["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    num_scale_targets = ["tenure", "MonthlyCharges", "TotalCharges", "charge_per_tenure"]
    scaler = StandardScaler()
    X_train[num_scale_targets] = scaler.fit_transform(X_train[num_scale_targets])
    X_test[num_scale_targets]  = scaler.transform(X_test[num_scale_targets])
    
    smote_engine = SMOTE(random_state=42)
    X_resampled, y_resampled = smote_engine.fit_resample(X_train, y_train)
    
    @st.cache_resource
    def compile_production_model(_X_r, _y_r):
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(_X_r, _y_r)
        return clf

    rf_clf = compile_production_model(X_resampled, y_resampled)
    predictions = rf_clf.predict(X_test)
    prediction_probabilities = rf_clf.predict_proba(X_test)[:, 1]
    
    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Optimized Model F1-Score", f"{f1_score(y_test, predictions):.4f}")
    metric_col_2.metric("Area Under ROC Curve (AUC)", f"{roc_auc_score(y_test, prediction_probabilities):.4f}")
    
    # Importance Plot
    fig_importance, ax_importance = plt.subplots(figsize=(10, 3.8))
    feature_keys = X_resampled.columns.tolist()
    importance_frame = pd.DataFrame({"Feature": feature_keys, "Weight": rf_clf.feature_importances_})
    importance_frame = importance_frame.sort_values("Weight", ascending=False).head(10)
    
    ax_importance.barh(importance_frame["Feature"][::-1], importance_frame["Weight"][::-1], color=BLUE, edgecolor="white")
    ax_importance.set_title("Top 10 Feature Weights Identifying Active Risk Profiles")
    st.pyplot(fig_importance)
    
    # ── Phase 6: Operational Directives ──────────────────────────
    st.header("🎯 Section 6 – Strategic Interventions")
    st.info("Based on the generated importance rankings, the retention department should implement these steps:")
    st.markdown("""
    1. **Contract Migration Subsidies:** Offer billing point incentives to shift Month-to-Month users into structured annual cycles.
    2. **Onboarding Support Optimization:** Establish dedicated check-ins during the critical 0-12 month account registration lifecycle window.
    3. **Automated Clearing Incentives:** Offer a small line item discount to convert high-risk electronic check users onto recurring automated payment channels.
    """)

except Exception as global_err:
    st.error(f"Execution Error encountered during app run: {global_err}")