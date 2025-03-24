import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("💰 Loan Fraud Detection App")

# Sidebar
app_mode = st.sidebar.selectbox("Select mode", ["EDA", "Prediction"])

# EDA Section
if app_mode == "EDA":
    st.subheader("Exploratory Data Analysis (EDA)")
    uploaded_file = st.file_uploader("Upload CSV for EDA", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### First 5 Rows of the Dataset:")
        st.write(data.head())

        st.write("### Dataset Shape:")
        st.write(data.shape)

        st.write("### Data Types:")
        st.write(data.dtypes)

        st.write("### Null Value Check:")
        st.write(data.isnull().sum())

        st.write("### Target Value Counts (if column 'fraud' exists):")
        if 'fraud' in data.columns:
            st.write(data['fraud'].value_counts())
        else:
            st.warning("The dataset does not contain a 'fraud' column.")

        st.write("### Correlation Heatmap:")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

# Prediction Section
elif app_mode == "Prediction":
    st.subheader("Fraud Prediction")
    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())

        if 'fraud' in data.columns:
            # Map target if required
            if data['fraud'].dtype == 'object':
                data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0})

            # Drop rows with missing target
            data = data.dropna(subset=['fraud'])

            # Split features and labels
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # Fill missing feature values
            X = X.fillna(0)

            # One-hot encode if needed
            X = pd.get_dummies(X)

            # Check for both classes
            if len(y.unique()) < 2:
                st.error("❌ The target column 'fraud' must contain both classes (0 and 1).")
            elif len(X) == 0:
                st.error("❌ Dataset is empty after processing.")
            else:
                # Train/Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Train RandomForest model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Model Evaluation
                y_pred = model.predict(X_test)
                st.write("### Model Performance:")
                st.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))

                # Confusion matrix
                st.write("### Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # Prediction input form
                st.write("---")
                st.write("### Predict Fraud for New Inputs:")
                input_data = {}
                for col in X.columns:
                    input_data[col] = st.number_input(f"Enter {col}:", value=0.0)
                
                if st.button("Predict"):
                    input_df = pd.DataFrame([input_data])
                    input_df = input_df.reindex(columns=X.columns, fill_value=0)
                    prediction = model.predict(input_df)[0]
                    result = "🚨 Fraud Detected!" if prediction == 1 else "✅ No Fraud Detected."
                    st.success(result)
        else:
            st.warning("⚠️ The dataset does not contain a 'fraud' column.")

