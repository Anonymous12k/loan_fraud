import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    This is a machine learning-powered Loan Fraud Detection web app built using **Streamlit**.
    
    - Upload your loan dataset
    - Train and visualize
    - Make predictions whether a loan is fraudulent or not
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.write("## Upload your dataset for fraud detection")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())

        # Simple check if "fraud" or target column exists
        if 'fraud' not in data.columns:
            st.warning("⚠️ The dataset does not contain a 'fraud' column for prediction.")
        else:
            # Split into features and target
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display metrics
            st.write("### Model Performance")
            st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Prediction input
            st.write("---")
            st.write("### Predict Fraud for New Data")
            input_data = {}
            for col in X.columns:
                val = st.number_input(f"Enter {col}:", value=0.0)
                input_data[col] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f"✅ Prediction: {'Fraud Detected' if prediction == 1 else 'Not Fraudulent'}")

# Exploratory Data Analysis
elif app_mode == "Exploratory Data Analysis":
    st.write("## Upload your dataset for EDA")
    uploaded_file_eda = st.file_uploader("Upload a CSV file for EDA", type=["csv"])

    if uploaded_file_eda is not None:
        data_eda = pd.read_csv(uploaded_file_eda)
        st.write("### Dataset Overview")
        st.write(data_eda.head())

        st.write("### Dataset Info")
        st.write(data_eda.describe())

        # Plot distributions
        if 'fraud' in data_eda.columns:
            st.write("### Fraud Distribution")
            st.bar_chart(data_eda['fraud'].value_counts())

            st.write("### Correlation Heatmap")
            corr = data_eda.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr, cmap='coolwarm', interpolation='nearest')
            ax.figure.colorbar(im, ax=ax)
            st.pyplot(fig)

# Footer
st.sidebar.write("---")
st.sidebar.write("Made with ❤️ by Your Name")
