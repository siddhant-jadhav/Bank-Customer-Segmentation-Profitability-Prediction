import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Customer Segmentation",
    page_icon="🏦",
    layout="wide",
)


# ──────────────────────────────────────────────
# Caching: load data + train models ONCE
# ──────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    """Load dataset, clean, cluster, and train the logistic-regression model."""

    # --- 1. Data Loading ---
    try:
        df = pd.read_csv("bank_transactions.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n_samples = 1500
        df = pd.DataFrame(
            {
                "CustomerDOB": pd.to_datetime(
                    np.random.choice(
                        pd.date_range("1940-01-01", "2004-12-31"), n_samples
                    )
                ).strftime("%d/%m/%y"),
                "CustGender": np.random.choice(["M", "F"], n_samples),
                "CustAccountBalance": np.random.uniform(10, 5_000_000, n_samples).round(2),
                "TransactionAmount (INR)": np.random.uniform(10, 50_000, n_samples).round(2),
            }
        )
        df.to_csv("bank_transactions.csv", index=False)

    # --- 2. Data Cleaning ---
    if "CustomerDOB" in df.columns:
        df["CustomerDOB"] = pd.to_datetime(df["CustomerDOB"], format="%d/%m/%y", errors="coerce")
        df.loc[df["CustomerDOB"].dt.year > 2023, "CustomerDOB"] -= pd.DateOffset(years=100)
        df["Age"] = 2016 - df["CustomerDOB"].dt.year
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df.drop(columns=["CustomerDOB"], inplace=True, errors="ignore")

    if df["CustGender"].dtype == object:
        df["CustGender"] = df["CustGender"].map({"M": 1, "F": 0, "T": 1}).fillna(1)
    else:
        df["CustGender"] = df["CustGender"].fillna(1)

    df["CustAccountBalance"] = df["CustAccountBalance"].fillna(df["CustAccountBalance"].median())
    df["TransactionAmount (INR)"] = df["TransactionAmount (INR)"].fillna(
        df["TransactionAmount (INR)"].median()
    )

    # --- 3. K-Means Clustering ---
    cluster_features = ["Age", "CustAccountBalance", "TransactionAmount (INR)"]
    X_cluster = df[cluster_features]

    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

    # --- 4. Logistic Regression ---
    df["HighProfit"] = (df["CustAccountBalance"] > 2_000_000).astype(int)

    features = ["Age", "CustGender", "CustAccountBalance", "TransactionAmount (INR)"]
    X_pred = df[features]
    y_pred_target = df["HighProfit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_pred, y_pred_target, test_size=0.2, random_state=42, stratify=y_pred_target
    )

    lr_scaler = StandardScaler()
    X_train_scaled = lr_scaler.fit_transform(X_train)
    X_test_scaled = lr_scaler.transform(X_test)

    # Removed class_weight='balanced' — it was causing too many false positives
    # on this heavily imbalanced dataset (the model was aggressively over-
    # predicting High Profit for customers who are actually Standard).
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # --- 5. Compute cluster-level stats for predictions page ---
    cluster_stats = df.groupby("Cluster").agg(
        avg_age=("Age", "mean"),
        avg_balance=("CustAccountBalance", "mean"),
        avg_txn=("TransactionAmount (INR)", "mean"),
        count=("Cluster", "size"),
        high_profit_pct=("HighProfit", "mean"),
    ).round(2)
    cluster_stats["high_profit_pct"] = (cluster_stats["high_profit_pct"] * 100).round(1)

    return {
        "df": df,
        "model": model,
        "lr_scaler": lr_scaler,
        "cluster_scaler": cluster_scaler,
        "features": features,
        "cluster_features": cluster_features,
        "kmeans": kmeans,
        "accuracy": accuracy,
        "cm": cm,
        "report": report,
        "auc_score": auc_score,
        "fpr": fpr,
        "tpr": tpr,
        "y_test": y_test,
        "y_pred": y_pred,
        "cluster_stats": cluster_stats,
    }


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
def predict_new_customer(data_dict, model, scaler, features):
    """Return (prediction, probability%) for a single customer."""
    user_df = pd.DataFrame([data_dict], columns=features)
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]
    return int(prediction), probability * 100


def assign_cluster(customer_data, kmeans, cluster_scaler, cluster_features):
    """Assign a single customer to a K-Means cluster."""
    # customer_data should be [Age, CustAccountBalance, TransactionAmount]
    user_df = pd.DataFrame([customer_data], columns=cluster_features)
    user_scaled = cluster_scaler.transform(user_df)
    cluster_id = kmeans.predict(user_scaled)[0]
    return int(cluster_id)


# ──────────────────────────────────────────────
# Load everything
# ──────────────────────────────────────────────
bundle = load_and_train()
df = bundle["df"]
model = bundle["model"]
lr_scaler = bundle["lr_scaler"]
features = bundle["features"]

# ──────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────
st.sidebar.title("🏦 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📊 EDA",
        "🧩 Clustering",
        "🔮 Predict",
        "📈 Model Evaluation",
    ],
)

# ──────────────────────────────────────────────
# Page: Overview
# ──────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏦 Retail Bank Customer Segmentation & Profitability Prediction")
    st.markdown("**Project by Group 6**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features Used", len(features))
    col3.metric("Clusters (K)", 3)
    col4.metric("Model Accuracy", f"{bundle['accuracy']*100:.2f}%")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Quick Stats")
    st.dataframe(df.describe().round(2), width="stretch")

# ──────────────────────────────────────────────
# Page: EDA
# ──────────────────────────────────────────────
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Age"], bins=30, kde=True, ax=ax1, color="#4C72B0")
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

    with col2:
        st.subheader("Account Balance Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df["CustAccountBalance"], bins=30, ax=ax2, color="#DD8452")
        ax2.set_xlabel("Balance (INR)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Transaction Amount Distribution")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.histplot(df["TransactionAmount (INR)"], bins=30, kde=True, ax=ax3, color="#55A868")
        ax3.set_xlabel("Amount (INR)")
        ax3.set_ylabel("Frequency")
        st.pyplot(fig3)

    with col4:
        st.subheader("Gender Split")
        gender_counts = df["CustGender"].value_counts()
        labels = ["Male" if g == 1 else "Female" for g in gender_counts.index]
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.pie(gender_counts, labels=labels, autopct="%1.1f%%", startangle=90,
                colors=["#4C72B0", "#DD8452"])
        ax4.set_title("Gender Distribution")
        st.pyplot(fig4)

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

# ──────────────────────────────────────────────
# Page: Clustering
# ──────────────────────────────────────────────
elif page == "🧩 Clustering":
    st.title("🧩 Customer Segmentation (K-Means)")
    st.markdown("---")

    cluster_names = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}

    col1, col2, col3 = st.columns(3)
    for idx, col in enumerate([col1, col2, col3]):
        subset = df[df["Cluster"] == idx]
        with col:
            st.markdown(f"### {cluster_names[idx]}")
            st.metric("Customers", f"{len(subset):,}")
            st.metric("Avg Age", f"{subset['Age'].mean():.1f}")
            st.metric("Avg Balance", f"₹{subset['CustAccountBalance'].mean():,.0f}")
            st.metric("Avg Txn", f"₹{subset['TransactionAmount (INR)'].mean():,.0f}")

    st.markdown("---")

    x_axis = st.selectbox("X-axis", bundle["cluster_features"], index=0)
    y_axis = st.selectbox("Y-axis", bundle["cluster_features"], index=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df[x_axis],
        df[y_axis],
        c=df["Cluster"],
        cmap="viridis",
        alpha=0.5,
        s=10,
    )
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Clusters: {x_axis} vs {y_axis}")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    st.subheader("Cluster Statistics")
    st.dataframe(
        df.groupby("Cluster")[bundle["cluster_features"]].describe().round(2),
        width="stretch",
    )

# ──────────────────────────────────────────────
# Page: Predict
# ──────────────────────────────────────────────
elif page == "🔮 Predict":
    st.title("🔮 Predict Customer Profitability")
    st.markdown("Enter customer details below to get a **profitability prediction**, **cluster assignment**, and **risk assessment**.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        balance = st.number_input(
            "Account Balance (INR)", min_value=0.0, value=500000.0, step=10000.0, format="%.2f"
        )
        txn_amount = st.number_input(
            "Transaction Amount (INR)", min_value=0.0, value=5000.0, step=100.0, format="%.2f"
        )

    gender_code = 1.0 if gender == "Male" else 0.0

    if st.button("🔍 Predict", use_container_width=True, type="primary"):
        # --- Profitability prediction ---
        prediction, prob = predict_new_customer(
            [age, gender_code, balance, txn_amount],
            model,
            lr_scaler,
            features,
        )

        # --- Cluster assignment ---
        cluster_id = assign_cluster(
            [age, balance, txn_amount],
            bundle["kmeans"],
            bundle["cluster_scaler"],
            bundle["cluster_features"],
        )

        is_high_profit = prediction == 1
        status_label = "HIGH PROFIT 🟢" if is_high_profit else "STANDARD 🔵"

        st.markdown("---")
        st.subheader("📋 Results")

        # ---- Top metrics row ----
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", status_label)
        m2.metric("High Profit Probability", f"{prob:.2f}%")
        m3.metric("Customer Cluster", f"Cluster {cluster_id}")

        st.progress(min(prob / 100.0, 1.0))

        # ---- Detailed analysis ----
        st.markdown("---")
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.subheader("🎯 Profitability Assessment")
            if is_high_profit:
                st.success(f"This customer is predicted to be **High Profit** with **{prob:.1f}%** confidence.")
                st.markdown("""
                **Recommendations:**
                - 🏆 Assign a dedicated relationship manager
                - 💳 Offer premium banking products & credit cards
                - 📊 Monitor for cross-sell opportunities (investments, insurance)
                - ⭐ Enroll in VIP loyalty program
                """)
            else:
                st.info(f"This customer is predicted to be **Standard** with **{100 - prob:.1f}%** confidence.")
                st.markdown("""
                **Recommendations:**
                - 📈 Encourage higher deposits with attractive interest rates
                - 💰 Promote savings & recurring deposit schemes
                - 🎁 Offer targeted promotions to increase account balance
                - 📱 Promote digital banking features for engagement
                """)

        with detail_col2:
            st.subheader("📊 Cluster Comparison")
            cluster_stats = bundle["cluster_stats"]
            c_stats = cluster_stats.loc[cluster_id]

            comparison_data = {
                "Metric": ["Age", "Account Balance (₹)", "Txn Amount (₹)"],
                "This Customer": [f"{age}", f"{balance:,.0f}", f"{txn_amount:,.0f}"],
                "Cluster Avg": [
                    f"{c_stats['avg_age']:.1f}",
                    f"{c_stats['avg_balance']:,.0f}",
                    f"{c_stats['avg_txn']:,.0f}",
                ],
            }
            st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)

            st.markdown(f"""
            **Cluster {cluster_id} Profile:**
            - 👥 **{int(c_stats['count']):,}** customers in this cluster
            - 💎 **{c_stats['high_profit_pct']:.1f}%** are High Profit
            """)

        # ---- Risk level ----
        st.markdown("---")
        st.subheader("⚠️ Risk Assessment")

        if balance < 10_000:
            risk = "HIGH"
            risk_color = "🔴"
            risk_msg = "Very low account balance. High risk of churn. Immediate engagement recommended."
        elif balance < 100_000:
            risk = "MEDIUM-HIGH"
            risk_color = "🟠"
            risk_msg = "Below-average balance. Customer may need incentives to stay active."
        elif balance < 500_000:
            risk = "MEDIUM"
            risk_color = "🟡"
            risk_msg = "Average customer profile. Standard retention strategies apply."
        elif balance < 2_000_000:
            risk = "LOW"
            risk_color = "🟢"
            risk_msg = "Healthy balance. Low churn risk. Good candidate for premium products."
        else:
            risk = "VERY LOW"
            risk_color = "💚"
            risk_msg = "High-value customer. Minimal churn risk. Prioritize relationship management."

        r1, r2 = st.columns([1, 3])
        r1.metric("Risk Level", f"{risk_color} {risk}")
        r2.markdown(f"**Assessment:** {risk_msg}")

# ──────────────────────────────────────────────
# Page: Model Evaluation
# ──────────────────────────────────────────────
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{bundle['accuracy']*100:.2f}%")
    col2.metric("AUC Score", f"{bundle['auc_score']:.4f}")

    report = bundle["report"]
    if "1" in report:
        col3.metric("F1 (High Profit)", f"{report['1']['f1-score']:.4f}")

    st.markdown("---")

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            bundle["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Standard", "High Profit"],
            yticklabels=["Standard", "High Profit"],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    with eval_col2:
        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(
            bundle["fpr"],
            bundle["tpr"],
            color="orange",
            lw=2,
            label=f"ROC Curve (AUC = {bundle['auc_score']:.2f})",
        )
        ax_roc.plot([0, 1], [0, 1], color="navy", linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), width="stretch")
