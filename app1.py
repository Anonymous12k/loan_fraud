import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🎨 Set Streamlit app title
st.title("🔍 Loan Fraud Detection System")

# 📝 Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # 📂 Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("✅ **Dataset Loaded Successfully!**")
    
    # 📊 Show dataset preview
    if st.checkbox("Show first 5 rows of data"):
        st.write(data.head())

    # 📈 Exploratory Data Analysis (EDA)
    if st.checkbox("Perform EDA"):
        st.subheader("📊 Data Insights")
        st.write(f"Total Rows: {data.shape[0]}, Total Columns: {data.shape[1]}")
        st.write("📌 **Missing Values:**")
        st.write(data.isnull().sum())

        st.write("📌 **Target Column Distribution (Fraud/Not Fraud):**")
        if 'fraud' in data.columns:
            st.bar_chart(data['fraud'].value_counts())
        else:
            st.error("❌ No 'fraud' column found in dataset!")

        # Correlation heatmap
        st.subheader("📊 Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

    # 🎯 Prediction Section
    if 'fraud' in data.columns:
        st.subheader("🤖 Train Model & Predict Loan Fraud")

        # ✅ Data Preprocessing
        data = data.dropna(subset=['fraud'])
        data = data[data['fraud'].isin([0, 1])]
        data['fraud'] = data['fraud'].astype(int)

        # 💡 Separate Features & Target
        X = data.drop('fraud', axis=1)
        X = pd.get_dummies(X)  # Convert categorical to numeric
        X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinities
        X = X.fillna(0)  # Fill missing values

        y = data['fraud']

        st.write(f"✅ **Features Shape:** {X.shape}, **Target Size:** {len(y)}")

        if len(y) > 0:
            # 🏋️‍♂️ Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # 🌲 Train Model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 📊 Evaluate Model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("### ✅ Model Performance")
            st.write(f"🎯 Accuracy: **{accuracy:.2f}**")
            st.text("📊 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # 🔎 Predict on User Input
            st.write("---")
            st.write("### 🔍 Predict Fraud for New Transaction")
            input_data = {}
            for col in X.columns:
                input_data[col] = st.number_input(f"Enter value for {col}:", value=0.0)

            if st.button("🔮 Predict"):
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                result = "⚠️ Fraud Detected!" if prediction == 1 else "✅ Not Fraudulent"
                st.success(f"Prediction: {result}")
        else:
            st.warning("⚠️ No valid data after cleaning! Please check your dataset.")
    else:
        st.error("❌ The dataset must contain a 'fraud' column for training.")

else:
    st.warning("📂 Please upload a dataset to proceed.")
