import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Page configuration
st.set_page_config(page_title="Loan Fraud Detection", page_icon="💰")

st.title("💰 Loan Fraud Detection App")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Mode", ["About", "Prediction", "Exploratory Data Analysis"])

if app_mode == "About":
    st.write("""
    ## About this App
    This app uses Machine Learning to detect loan fraud.
    
    ✅ Upload your dataset  
    ✅ The app will train a model and make predictions  
    ✅ Visualize dataset insights  
    """)

elif app_mode == "Prediction":
    st.write("## Upload your dataset for fraud prediction")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())

        st.write("### Missing Values Check")
        st.write(data.isnull().sum())

        # Check and convert 'fraud' column
        if 'fraud' in data.columns:
            data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0}).fillna(data['fraud'])
            data = data.dropna(subset=['fraud'])
            data['fraud'] = pd.to_numeric(data['fraud'], errors='coerce')
            data = data.dropna(subset=['fraud'])

            # Fill missing values for features
            data = data.fillna(0)

            # Features and labels
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Convert categorical columns to dummies
            X = pd.get_dummies(X)

            st.write(f"✅ Features shape: {X.shape}, Labels length: {len(y)}")
            st.write(f"✅ Target value counts:\n{y.value_counts()}")

            # Validation checks before splitting
            if X.empty or y.empty:
                st.error("❌ The data is empty after cleaning. Please check the dataset.")
            elif len(X) != len(y):
                st.error("❌ Mismatch between feature rows and target values.")
            elif y.nunique() < 2:
                st.error("❌ Target column must contain at least two classes.")
            elif len(X) < 10:
                st.error("❌ Dataset too small. Please provide more data.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Train Random Forest
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Model results
                st.write("### ✅ Model Performance")
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                # Prediction on user input
                st.write("### Predict on New Data")
                input_data = {}
                for feature in X.columns:
                    input_data[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

                if st.button("Predict Fraud"):
                    new_input_df = pd.DataFrame([input_data])
                    prediction = model.predict(new_input_df)[0]
                    st.success(f"✅ Prediction: {'Fraud Detected' if prediction == 1 else 'No Fraud Detected'}")
        else:
            st.warning("❗ The dataset does not contain a 'fraud' column.")

elif app_mode == "Exploratory Data Analysis":
    st.write("## Upload your dataset for EDA")
    uploaded_eda = st.file_uploader("Upload CSV file for EDA", type=["csv"])

    if uploaded_eda is not None:
        data_eda = pd.read_csv(uploaded_eda)
        st.write("### Dataset Head")
        st.dataframe(data_eda.head())

        st.write("### Dataset Description")
        st.write(data_eda.describe())

        st.write("### Missing Values Count")
        st.write(data_eda.isnull().sum())

        # Correlation heatmap
        st.write("### Correlation Heatmap (Numerical Columns Only)")
        num_data = data_eda.select_dtypes(include=[np.number])
        if not num_data.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(num_data.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for correlation heatmap.")

st.sidebar.write("---")
st.sidebar.write("Developed by Nithish S")

