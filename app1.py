import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")
st.title("💰 Loan Fraud Detection App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    if 'fraud' in data.columns:
        # Convert fraud column to binary if needed
        if data['fraud'].dtype == 'object':
            data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0})
        
        # Fill all missing values
        data = data.fillna(0)

        # Separate features and target
        X = data.drop('fraud', axis=1)
        y = data['fraud']

        # Convert categorical columns into dummy variables
        X = pd.get_dummies(X)

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Model Accuracy")
        st.write(f"{accuracy_score(y_test, y_pred)*100:.2f}%")

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("## Predict on New Data")
        input_data = {}
        for feature in X.columns:
            input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
        
        if st.button("Predict Fraud"):
            new_input_df = pd.DataFrame([input_data])
            prediction = model.predict(new_input_df)[0]
            result = "🚨 Fraud Detected!" if prediction == 1 else "✅ No Fraud Detected."
            st.success(result)
    else:
        st.error("The dataset does not contain a 'fraud' column.")

st.markdown("---")
st.caption("Made with ❤️ by Nithish S")
