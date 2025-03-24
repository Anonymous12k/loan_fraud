import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ✅ Set Streamlit page config
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")

# App Title
st.title("💰 Loan Fraud Detection App")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select mode", ["About", "EDA", "Prediction"])

# ✅ About Page
if app_mode == "About":
    st.subheader("📖 About This App")
    st.write("""
    Welcome to the **Loan Fraud Detection** app built with **Streamlit**.  

    ✅ What you can do here:
    - Upload your loan dataset  
    - Perform Exploratory Data Analysis (EDA)  
    - Train a Random Forest model and make predictions  
    - Predict if a new loan transaction is fraudulent or not  

    🔎 Technologies used:
    - Python, Pandas, Scikit-learn, Streamlit, Seaborn, Matplotlib  

    🚀 Developed for data science learning and practical demonstration.
    """)

# ✅ EDA Page
elif app_mode == "EDA":
    st.subheader("📊 Exploratory Data Analysis")
    uploaded_file_eda = st.file_uploader("Upload a CSV file for EDA", type=["csv"])

    if uploaded_file_eda is not None:
        data_eda = pd.read_csv(uploaded_file_eda)
        st.write("### Dataset Overview")
        st.dataframe(data_eda.head())

        st.write("### Dataset Information")
        st.write(data_eda.describe())

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        numeric_data = data_eda.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available for correlation heatmap.")

# ✅ Prediction Page
elif app_mode == "Prediction":
    st.subheader("🔎 Predict Loan Fraud")
    uploaded_file = st.file_uploader("Upload your dataset for prediction", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        if 'fraud' not in data.columns:
            st.error("❌ The dataset must contain a 'fraud' column for training.")
        else:
            # Convert 'fraud' column to binary if needed
            if data['fraud'].dtype == 'object':
                data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0})

            # Split features & target
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Convert categorical columns
            X = pd.get_dummies(X)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train the Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Show model performance
            st.write("### ✅ Model Performance")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.write("---")
            st.write("### ➡️ Predict Fraud for New Transaction")
            input_data = {}
            for col in X.columns:
                input_data[col] = st.number_input(f"Enter value for {col}:", value=0.0)

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                result = "Fraud Detected ⚠️" if prediction == 1 else "Not Fraudulent ✅"
                st.success(f"Prediction: {result}")

# ✅ Footer
st.sidebar.write("---")
st.sidebar.write("👨‍💻 Developed by Nithish S")
