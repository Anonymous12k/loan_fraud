import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ Must be the first Streamlit call
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")

# App Title
st.title("💰 Loan Fraud Detection App")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Prediction", "Exploratory Data Analysis"])

# About Page
if app_mode == "About":
    st.write("""
    ## About
    This is a machine learning-powered **Loan Fraud Detection** web app built using **Streamlit**.
    
    - Upload your loan dataset
    - Automatically train a Random Forest model
    - Make predictions on new input data
    - Visualize fraud distribution and correlations
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.write("## Upload your dataset for fraud detection")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())

        # Handle missing values
        if data.isnull().sum().sum() > 0:
            st.warning("Missing values found. Filling numeric columns with median.")
            data = data.fillna(data.median(numeric_only=True))

        # Encode fraud column if it has Yes/No
        if 'fraud' in data.columns:
            if data['fraud'].dtype == 'object':
                data['fraud'] = data['fraud'].map({'Yes': 1, 'No': 0})

            # Separate features and target
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Encode categorical columns
            X_encoded = pd.get_dummies(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model Performance
            st.write("### Model Performance")
            st.write(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Prediction input form
            st.write("---")
            st.write("### Predict Fraud for New Data")
            input_data = {}
            st.write("Enter values for the following features:")
            for col in X_encoded.columns:
                val = st.number_input(f"{col}:", value=0.0)
                input_data[col] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                if prediction == 1:
                    st.error("⚠️ Prediction: Fraud Detected!")
                else:
                    st.success("✅ Prediction: Not Fraudulent")
        else:
            st.warning("⚠️ The dataset does not contain a 'fraud' column for prediction.")

# Exploratory Data Analysis
elif app_mode == "Exploratory Data Analysis":
    st.write("## Upload your dataset for EDA")
    uploaded_file_eda = st.file_uploader("Upload a CSV file for EDA", type=["csv"])

    if uploaded_file_eda is not None:
        data_eda = pd.read_csv(uploaded_file_eda)
        st.write("### Dataset Overview")
        st.write(data_eda.head())

        st.write("### Dataset Description")
        st.write(data_eda.describe())

        # Handle missing values
        if data_eda.isnull().sum().sum() > 0:
            st.warning("Missing values found. Displaying columns with missing values:")
            st.write(data_eda.isnull().sum()[data_eda.isnull().sum() > 0])

        # Fraud distribution chart
        if 'fraud' in data_eda.columns:
            if data_eda['fraud'].dtype == 'object':
                data_eda['fraud'] = data_eda['fraud'].map({'Yes': 1, 'No': 0})
            st.write("### Fraud Distribution")
            st.bar_chart(data_eda['fraud'].value_counts())

            # Correlation Heatmap
            st.write("### Correlation Heatmap")
            corr = data_eda.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ 'fraud' column not found for distribution and correlation analysis.")

# Footer
st.sidebar.write("---")
st.sidebar.write("Developed by Nithish S")
