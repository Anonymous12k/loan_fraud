import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set page config
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")

st.title("💰 Loan Fraud Detection App")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Please select an operation", "Prediction", "Exploratory Data Analysis"])

# ------------------------- About -------------------------
if app_mode == "Please select an operation":
    st.write("""
    ## 
    This machine learning-powered **Loan Fraud Detection App** is built using **Streamlit**.
    
    ✔ Upload your loan dataset  
    ✔ Automatically train a Random Forest model  
    ✔ Visualize statistics  
    ✔ Predict if a loan is fraudulent based on new inputs  
    """)

# ------------------------- Prediction -------------------------
elif app_mode == "Prediction":
    st.write("## Upload your dataset for fraud detection")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        if 'fraud' in data.columns:
            # Clean data
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna(subset=['fraud'])  # Drop rows with NaN fraud labels
            if data['fraud'].dtype == 'object':
                data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0})

            # Separate features & target
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Handle categorical variables
            X = pd.get_dummies(X)

            # Align columns for safety
            X, y = X.align(y, join="inner", axis=0)

            # Make sure y doesn't have NaNs
            y = y.fillna(0).astype(int)

            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            st.write("### Model Performance")
            st.write(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # New predictions
            st.write("---")
            st.write("### Predict Fraud for New Data")
            st.info("Enter values below:")

            input_data = {}
            for col in X.columns:
                val = st.number_input(f"{col}:", value=0.0)
                input_data[col] = val

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])

                # Add missing columns if any
                missing_cols = set(X.columns) - set(input_df.columns)
                for c in missing_cols:
                    input_df[c] = 0
                input_df = input_df[X.columns]  # Align order

                prediction = model.predict(input_df)[0]
                st.success(f"✅ Prediction: {'Fraud Detected' if prediction == 1 else 'Not Fraudulent'}")
        else:
            st.warning("⚠️ The dataset does not contain a 'fraud' column. Please upload a correct dataset.")

# ------------------------- EDA -------------------------
elif app_mode == "Exploratory Data Analysis":
    st.write("## Upload your dataset for EDA")
    uploaded_file_eda = st.file_uploader("Upload a CSV file for EDA", type=["csv"])

    if uploaded_file_eda is not None:
        data_eda = pd.read_csv(uploaded_file_eda)
        st.write("### Dataset Overview")
        st.dataframe(data_eda.head())

        st.write("### Statistical Description")
        st.write(data_eda.describe())

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
