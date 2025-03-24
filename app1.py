import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ✅ Set page config
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")

# App Title
st.title("💰 Loan Fraud Detection App")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["About", "Prediction", "Exploratory Data Analysis"])

# ----------------- About Page -----------------
if app_mode == "About":
    st.write("""
    ## About  
    This is a machine learning-powered **Loan Fraud Detection web app** built using **Streamlit**.  
    
    - Upload your loan dataset  
    - The app will train a model and visualize the data  
    - Predict whether a loan is fraudulent or not by entering new data  
    """)

# ----------------- Prediction Page -----------------
elif app_mode == "Prediction":
    st.write("## Upload your dataset for fraud detection")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        # Ensure fraud column is present and mapped
        if 'fraud' in data.columns:
            if data['fraud'].dtype == 'object':
                data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0})

            # Split data into features and target
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Handle categorical columns
            X = pd.get_dummies(X)

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display model performance
            st.write("### Model Performance")
            st.write(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Prediction input form
            st.write("---")
            st.write("### Predict Fraud for New Data")
            st.info("Fill out the fields below:")

            input_data = {}
            for col in X.columns:
                if np.issubdtype(X[col].dtype, np.number):
                    input_value = st.number_input(f"{col}:", value=0.0)
                else:
                    input_value = st.text_input(f"{col}:")
                input_data[col] = input_value

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])

                # Add missing columns with default values (0)
                missing_cols = set(X.columns) - set(input_df.columns)
                for col in missing_cols:
                    input_df[col] = 0

                # Ensure correct column order
                input_df = input_df[X.columns]

                # Predict using trained model
                prediction = model.predict(input_df)[0]
                st.success(f"✅ Prediction: {'Fraud Detected' if prediction == 1 else 'Not Fraudulent'}")
        else:
            st.warning("⚠️ The dataset does not contain a 'fraud' column. Please upload the correct dataset.")

# ----------------- EDA Page -----------------
elif app_mode == "Exploratory Data Analysis":
    st.write("## Upload your dataset for EDA")
    uploaded_file_eda = st.file_uploader("Upload a CSV file for EDA", type=["csv"])

    if uploaded_file_eda is not None:
        data_eda = pd.read_csv(uploaded_file_eda)
        st.write("### Dataset Overview")
        st.dataframe(data_eda.head())

        st.write("### Dataset Description")
        st.write(data_eda.describe())

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        numeric_data = data_eda.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns found for correlation heatmap.")

# ----------------- Footer -----------------
st.sidebar.write("---")
st.sidebar.write("Developed by Nithish S")
