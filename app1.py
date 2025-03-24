import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
from datetime import datetime

# --- Cache the model & scaler loading for faster reloads ---
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["feature_names"]

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

model, feature_names = load_model()
scaler = load_scaler()

# --- Initialize SQLite Database ---
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_input TEXT,
    prediction TEXT,
    probability REAL
)
""")
conn.commit()

# --- Streamlit Frontend Setup ---
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")
st.title("💰 Loan Fraud Detection System")
st.write("Enter the applicant's details below to check for potential fraud.")

# --- User Input Form ---
input_fields = {
    "Age": st.number_input("Age", 18, 100),
    "Occupation": st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"]),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married"]),
    "Dependents": st.number_input("Dependents", 0, 10),
    "ResidentialStatus": st.selectbox("Residential Status", ["Owned", "Rented", "Living with Parents"]),
    "AddressDuration": st.number_input("Address Duration (Years)", 0, 50),
    "CreditScore": st.number_input("Credit Score", 300, 850),
    "IncomeLevel": st.number_input("Income Level", 0),
    "LoanAmountRequested": st.number_input("Loan Amount Requested", 0),
    "LoanTerm": st.number_input("Loan Term (Months)", 1, 360),
    "PurposeoftheLoan": st.selectbox("Purpose of Loan", ["Education", "Medical", "Home", "Business", "Other"]),
    "Collateral": st.selectbox("Collateral", ["Yes", "No"]),
    "InterestRate": st.number_input("Interest Rate (%)", 0.0),
    "PreviousLoans": st.number_input("Previous Loans", 0),
    "ExistingLiabilities": st.number_input("Existing Liabilities", 0),
    "ApplicationBehavior": st.selectbox("Application Behavior", ["Normal", "Aggressive"]),
    "LocationofApplication": st.text_input("Location of Application"),
    "ChangeinBehavior": st.selectbox("Change in Behavior", ["No", "Yes"]),
    "TimeofTransaction": st.text_input("Time of Transaction (HH:MM)", "10:00"),
    "AccountActivity": st.selectbox("Account Activity", ["Normal", "High", "Suspicious"]),
    "PaymentBehavior": st.selectbox("Payment Behavior", ["Regular", "Late"]),
    "Blacklists": st.selectbox("Blacklists", ["Yes", "No"]),
    "EmploymentVerification": st.selectbox("Employment Verification", ["Verified", "Not Verified"]),
    "PastFinancialMalpractices": st.selectbox("Past Financial Malpractices", ["Yes", "No"]),
    "DeviceInformation": st.selectbox("Device Information", ["Trusted", "Unknown"]),
    "SocialMediaFootprint": st.selectbox("Social Media Footprint", ["Strong", "Weak"]),
    "ConsistencyinData": st.selectbox("Consistency in Data", ["Consistent", "Inconsistent"]),
    "Referral": st.selectbox("Referral", ["Yes", "No"])
}

# --- Prediction Section ---
if st.button("🔎 Predict Loan Fraud"):
    input_df = pd.DataFrame([input_fields])
    input_encoded = pd.get_dummies(input_df).reindex(columns=feature_names, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! (Probability: {probability * 100:.2f}%)")
    else:
        st.success(f"✅ No Fraud Detected (Probability of fraud: {probability * 100:.2f}%)")

    # Store prediction record in DB
    cursor.execute("""
        INSERT INTO predictions (timestamp, user_input, prediction, probability)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps(input_fields), str(prediction), float(probability)))
    conn.commit()

# --- Sidebar for Dashboard & Analytics ---
st.sidebar.title("📊 Dashboard")
total_preds = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
fraud_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='1'").fetchone()[0]

st.sidebar.metric("Total Predictions", total_preds)
st.sidebar.metric("Frauds Detected", fraud_count)

if st.sidebar.button("📈 Show Daily Fraud Trend"):
    df = pd.read_sql("SELECT * FROM predictions", conn)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        daily_fraud = df[df['prediction'] == '1'].groupby('date').count()['id']
        st.sidebar.line_chart(daily_fraud)

if st.sidebar.button("📃 View Prediction Records"):
    df = pd.read_sql("SELECT * FROM predictions", conn)
    st.dataframe(df)

st.sidebar.markdown("---")
st.sidebar.write("Made with ❤️ by L KISHORE")

