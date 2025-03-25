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
            data = data.dropna(subset=['fraud'])
            if data['fraud'].dtype == 'object':
                data['fraud'] = data['fraud'].map({'yes': 1, 'no': 0})

            # Separate features & target
            X = data.drop('fraud', axis=1)
            y = data['fraud']

            # One-hot encode categorical columns
            X = pd.get_dummies(X)

            # Align columns and clean
            X, y = X.align(y, join="inner", axis=0)
            y = y.fillna(0).astype(int)

            # Split data and train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Show model performance
            st.write("### Model Performance")
            st.write(f"✅ Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Input for new prediction
            st.write("---")
            st.write("## Predict Fraud for New Data")
            st.info("Enter details below for prediction:")

            input_data = {
                'Age': st.number_input("Age", min_value=18, max_value=100, step=1),
                'Occupation': st.selectbox("Occupation", [
                    'Accountant', 'Architect', 'Artist', 'Chef', 'Clerk', 'Designer', 'Doctor', 
                    'Engineer', 'Entrepreneur', 'Lawyer', 'Manager', 'Musician', 'Retired', 
                    'Self-employed', 'Software Developer', 'Student', 'Teacher', 'Technician', 'Unemployed'
                ]),
                'MaritalStatus': st.selectbox("Marital Status", ['Divorced', 'Married', 'Single']),
                'Dependents': st.number_input("Dependents", min_value=0, step=1),
                'ResidentialStatus': st.selectbox("Residential Status", ['Own', 'Rent', 'Live with Parents']),
                'AddressDuration': st.number_input("Address Duration (years)", min_value=0, step=1),
                'CreditScore': st.number_input("Credit Score", min_value=300, max_value=850, step=1),
                'IncomeLevel': st.number_input("Income Level", min_value=0.0),
                'LoanAmountRequested': st.number_input("Loan Amount Requested", min_value=0.0),
                'LoanTerm': st.number_input("Loan Term (months)", min_value=1, step=1),
                'PurposeoftheLoan': st.selectbox("Purpose of the Loan", ['auto', 'education', 'home', 'medical', 'personal', 'travel']),
                'Collateral': st.selectbox("Collateral", ['Yes', 'No']),
                'InterestRate': st.number_input("Interest Rate (%)", min_value=0.0),
                'PreviousLoans': st.number_input("Previous Loans Taken", min_value=0, step=1),
                'ExistingLiabilities': st.number_input("Existing Liabilities", min_value=0.0),
                'ApplicationBehavior': st.selectbox("Application Behavior", ['Normal', 'Rapid']),
                'LocationofApplication': st.selectbox("Location of Application", ['Local', 'Unusual']),
                'ChangeinBehavior': st.selectbox("Change in Behavior", ['Yes', 'No']),
                'TimeofTransaction': st.text_input("Time of Transaction (e.g., 10:50)"),
                'AccountActivity': st.selectbox("Account Activity", ['Normal', 'Unusual']),
                'PaymentBehavior': st.selectbox("Payment Behavior", ['On-time', 'Late', 'Defaulted']),
                'Blacklists': st.selectbox("Blacklists", ['Yes', 'No']),
                'EmploymentVerification': st.selectbox("Employment Verification", ['Verified', 'Not Verified']),
                'PastFinancialMalpractices': st.selectbox("Past Financial Malpractices", ['Yes', 'No']),
                'DeviceInformation': st.selectbox("Device Used", ['Desktop', 'Laptop', 'Mobile', 'Tablet']),
                'SocialMediaFootprint': st.selectbox("Social Media Footprint", ['Yes', 'No']),
                'ConsistencyinData': st.selectbox("Consistency in Data", ['Consistent', 'Inconsistent']),
                'Referral': st.selectbox("Referral Method", ['Online', 'Referral'])
            }

            # Predict button
            if st.button("Predict"):
                # Convert categorical inputs into one-hot encoded form to match training data
                input_df = pd.DataFrame([input_data])
                input_df_encoded = pd.get_dummies(input_df)
                
                # Align columns with training data
                missing_cols = set(X.columns) - set(input_df_encoded.columns)
                for col in missing_cols:
                    input_df_encoded[col] = 0
                input_df_encoded = input_df_encoded[X.columns]
                
                # Make prediction
                prediction = model.predict(input_df_encoded)[0]
                result = 'Fraud Detected' if prediction == 1 else 'Not Fraudulent'
                st.success(f"✅ Prediction: {result}")

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
