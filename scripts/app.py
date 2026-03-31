# ============================================================
# JP Morgan Chase — AI Finance Agent
# Streamlit Web Application
# Author : Fabrice William FOMHOM
# Date   : March 2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import anthropic
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

# Load API key from Streamlit secrets
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "JP Morgan — AI Finance Agent",
    page_icon  = "🏦",
    layout     = "wide"
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(DATA_DIR, "jpmorgan_transactions.csv"),
        parse_dates=["date"]
    )

# ── Train ML model ────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    le      = LabelEncoder()
    df_ml   = df.copy()
    df_ml["category_encoded"]    = le.fit_transform(df_ml["category"])
    df_ml["merchant_encoded"]    = le.fit_transform(df_ml["merchant"])
    df_ml["day_of_week_encoded"] = le.fit_transform(df_ml["day_of_week"])

    FEATURES = ["amount","month","quarter",
                "category_encoded","merchant_encoded",
                "day_of_week_encoded"]

    X = df_ml[FEATURES]
    y = df_ml["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_train_b, y_train_b = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_b, y_train_b)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc         = roc_auc_score(y_test, y_pred_prob)

    return model, le, auc, FEATURES

# ── Load everything ───────────────────────────────────────────
df             = load_data()
model, le, auc, FEATURES = train_model(df)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/af/J_P_Morgan_Logo_2008_1.svg", width=200)
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", [
    "📊 Executive Dashboard",
    "🚨 Fraud Detection",
    "🤖 AI Agent Chat",
    "📄 Capstone Report"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** JP Morgan AI Finance Agent")
st.sidebar.markdown("**Author:** Fabrice William FOMHOM")
st.sidebar.markdown("**Date:** March 2026")

# ════════════════════════════════════════════════════════════
# PAGE 1 : EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "📊 Executive Dashboard":
    st.title("🏦 JP Morgan Chase — Executive Dashboard")
    st.markdown("**AI-Powered Financial Analytics | 2024**")
    st.markdown("---")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Volume",    f"${df['amount'].sum():,.0f}")
    col2.metric("Transactions",    f"{len(df):,}")
    col3.metric("Fraud Rate",      f"{df['is_fraud'].mean()*100:.2f}%",
                delta="-Target: <1%", delta_color="inverse")
    col4.metric("Fraud Losses",    f"${df[df['is_fraud']==1]['amount'].sum():,.0f}")
    col5.metric("ML AUC Score",    f"{auc:.4f}")

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Transaction Volume")
        monthly = df.groupby("month_name")["amount"].sum().reindex(
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(monthly.index, monthly.values,
               color="#003087", alpha=0.85)
        ax.set_ylabel("Amount ($)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Fraud Rate by Category")
        cat = df.groupby("category")["is_fraud"].mean().sort_values()
        colors = ["#D62728" if x > 0.03 else "#003087" for x in cat]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(cat.index, cat.values * 100, color=colors, alpha=0.85)
        ax.set_xlabel("Fraud Rate (%)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud vs Normal Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[df["is_fraud"]==0]["amount"], bins=40,
                color="#003087", alpha=0.7,
                label="Normal", density=True)
        ax.hist(df[df["is_fraud"]==1]["amount"], bins=40,
                color="#D62728", alpha=0.7,
                label="Fraud", density=True)
        ax.set_xlabel("Amount ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Top 10 Transactions")
        top10 = df.nlargest(10, "amount")[
            ["date","customer_id","merchant",
             "category","amount","is_fraud"]]
        top10["is_fraud"] = top10["is_fraud"].map(
            {0:"✅ Normal", 1:"🚨 FRAUD"})
        st.dataframe(top10, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 : FRAUD DETECTION
# ════════════════════════════════════════════════════════════
elif page == "🚨 Fraud Detection":
    st.title("🚨 Real-Time Fraud Detection")
    st.markdown("Enter a transaction to get an instant risk assessment.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        amount   = st.number_input("Transaction Amount ($)",
                                   min_value=1.0, max_value=15000.0,
                                   value=500.0, step=10.0)
        category = st.selectbox("Category",
                    df["category"].unique())
        merchant = st.selectbox("Merchant",
                    df["merchant"].unique())
        month    = st.slider("Month", 1, 12, 6)

        if st.button("🔍 Analyze Transaction", type="primary"):
            # Prepare features
            cat_enc = le.transform([category])[0] if category in le.classes_ else 0
            mer_enc = le.transform([merchant])[0] if merchant in le.classes_ else 0
            dow_enc = 0
            quarter = (month - 1) // 3 + 1

            features = np.array([[amount, month, quarter,
                                   cat_enc, mer_enc, dow_enc]])
            fraud_prob = model.predict_proba(features)[0][1]
            risk_score = int(fraud_prob * 100)

            with col2:
                st.subheader("Risk Assessment")
                if risk_score > 60:
                    st.error(f"🔴 HIGH RISK — Score: {risk_score}/100")
                    st.error("**RECOMMENDATION: BLOCK TRANSACTION**")
                elif risk_score > 30:
                    st.warning(f"🟡 MEDIUM RISK — Score: {risk_score}/100")
                    st.warning("**RECOMMENDATION: MONITOR CLOSELY**")
                else:
                    st.success(f"🟢 LOW RISK — Score: {risk_score}/100")
                    st.success("**RECOMMENDATION: APPROVE**")

                # Risk gauge
                fig, ax = plt.subplots(figsize=(6, 3))
                color = "#D62728" if risk_score>60 else \
                        "#FFA500" if risk_score>30 else "#28A745"
                ax.barh(["Risk Score"], [risk_score],
                        color=color, alpha=0.85)
                ax.barh(["Risk Score"], [100-risk_score],
                        left=[risk_score],
                        color="#e0e0e0", alpha=0.5)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Risk Score (0-100)")
                ax.axvline(x=30, color="orange",
                           linestyle="--", alpha=0.7)
                ax.axvline(x=60, color="red",
                           linestyle="--", alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown(f"""
                **Transaction Summary:**
                - Amount   : ${amount:,.2f}
                - Category : {category}
                - Merchant : {merchant}
                - Month    : {month}
                - Fraud Probability : {fraud_prob*100:.1f}%
                """)

# ════════════════════════════════════════════════════════════
# PAGE 3 : AI AGENT CHAT
# ════════════════════════════════════════════════════════════
elif page == "🤖 AI Agent Chat":
    st.title("🤖 JP Morgan AI Finance Agent")
    st.markdown("Chat with our AI analysts about your financial data.")
    st.markdown("---")

    # Agent selector
    agent_role = st.selectbox("Select Agent", [
        "🔍 Alex — Data Analyst",
        "⚠️ Sarah — Risk Supervisor",
        "👔 Michael — VP of Risk Management"
    ])

    # API Key input
    api_key = st.text_input("Anthropic API Key",
                             type="password",
                             placeholder="placeholder="sk-ant-api03-..."

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask your financial question..."):
        if not api_key:
            st.error("Please enter your API key above!")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            # Build context
            context = f"""
            JP Morgan Portfolio:
            - Transactions: {len(df):,}
            - Total Volume: ${df['amount'].sum():,.2f}
            - Fraud Rate: {df['is_fraud'].mean()*100:.2f}%
            - Fraud Losses: ${df[df['is_fraud']==1]['amount'].sum():,.2f}
            - Top fraud category: restaurant (7.58%)
            - ML Model AUC: {auc:.4f}
            """

            # Agent personalities
            personas = {
                "🔍 Alex — Data Analyst"          : "You are Alex, a detail-oriented Data Analyst at JP Morgan. Always cite specific numbers.",
                "⚠️ Sarah — Risk Supervisor"       : "You are Sarah, a Risk Supervisor at JP Morgan. Focus on risk implications.",
                "👔 Michael — VP of Risk Management": "You are Michael, VP at JP Morgan. Be strategic and concise."
            }

            with st.chat_message("assistant"):
                with st.spinner(f"Thinking..."):
                    client  = anthropic.Anthropic(api_key=api_key)
                    response = client.messages.create(
                        model      = "claude-sonnet-4-20250514",
                        max_tokens = 1024,
                        system     = personas[agent_role] + "\n\n" + context,
                        messages   = [{"role": "user",
                                       "content": prompt}]
                    )
                    reply = response.content[0].text
                    st.write(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply})

# ════════════════════════════════════════════════════════════
# PAGE 4 : CAPSTONE REPORT
# ════════════════════════════════════════════════════════════
elif page == "📄 Capstone Report":
    st.title("📄 Capstone Project Report")
    st.markdown("---")

    report_path = os.path.join(BASE_DIR, "docs", "capstone_report.txt")

    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = f.read()
        st.markdown(report)
        st.download_button(
            label    = "⬇️ Download Report",
            data     = report,
            file_name= "jpmorgan_capstone_report.txt",
            mime     = "text/plain"
        )
    else:
        st.warning("Report not found. Please run Lab 05 first.")
        
        
