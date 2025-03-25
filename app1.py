import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set page configuration
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")

# App title
st.title("💰 Loan Fraud Detection App")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Prediction", "Exploratory Data Analysis"])

# ------------------------- About Page -------------------------
if app_mode == "About":
    st.write("""
    ## About
    This machine learning-powered **Loan Fraud Detection Web App** is built using **Streamlit**.
    
    ✔ Upload your loan dataset  
    ✔ Automatically train a Random Forest model  
    ✔ Visualize dataset statistics  
    ✔ Predict whether a loan is fraudulent or not based on user input  
    """)

# ------------------------- Prediction Page -------------------------
elif app_mode == "Prediction":
    st.write("## Upload your dataset for fraud detection")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Check and convert 'fraud' column if present
        if 'fraud' in data.columns:
            if data['fraud'].dtype == 'object':
                data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0})

            # Separate features & target
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Handle categorical features
            X = pd.get_dummies(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Model performance metrics
            st.write("### Model Performance")
            st.write(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # New prediction
            st.write("---")
            st.write("### Predict Fraud for New Data")
            st.info("Please enter values for the following features:")

            input_data = {}
            for col in X.columns:
                if X[col].dtype in [np.float64, np.int64]:
                    val = st.number_input(f"{col}:", value=0.0)
                else:
                    val = st.text_input(f"{col}:")
                input_data[col] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])

                # Ensure columns match training set (missing columns get filled)
                missing_cols = set(X.columns) - set(input_df.columns)
                for c in missing_cols:
                    input_df[c] = 0
                input_df = input_df[X.columns]  # Order columns

                prediction = model.predict(input_df)[0]
                st.success(f"✅ Prediction: {'Fraud Detected' if prediction == 1 else 'Not Fraudulent'}")
        else:
            st.warning("⚠️ The dataset does not contain a 'fraud' column. Please upload a correct dataset.")

# ------------------------- EDA Page -------------------------
elif app_mode == "Exploratory Data Analysis":
    st.write("## Upload your dataset for Exploratory Data Analysis (EDA)")
    uploaded_file_eda = st.file_uploader("Upload a CSV file for EDA", type=["csv"])

    if uploaded_file_eda is not None:
        data_eda = pd.read_csv(uploaded_file_eda)
        st.write("### Dataset Overview")
        st.dataframe(data_eda.head())

        st.write("### Statistical Description")
        st.write(data_eda.describe())

        # Correlation heatmap visualization
        st.write("### Correlation Heatmap")
        numeric_data = data_eda.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available to plot correlation heatmap.")

# ------------------------- Footer -------------------------
st.sidebar.write("---")
st.sidebar.write("Developed by Nithish S")
