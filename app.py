import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Required for smooth deployment

# Configure app
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Bank Loan Approval Prediction")
st.write("""
Predict whether a loan application should be approved based on applicant details.
""")

# Generate sample data
@st.cache_data
def generate_loan_data():
    np.random.seed(42)
    n_applicants = 300
    
    data = pd.DataFrame({
        'Age': np.random.randint(18, 70, n_applicants),
        'Income': np.random.randint(20000, 150000, n_applicants),
        'LoanAmount': np.random.randint(5000, 300000, n_applicants),
        'CreditScore': np.random.randint(300, 850, n_applicants),
        'MonthsEmployed': np.random.randint(3, 120, n_applicants),
        'LoanTerm': np.random.choice([12, 24, 36, 60], n_applicants),
        'HasDefaulted': np.random.choice([0, 1], n_applicants, p=[0.8, 0.2])
    })
    
    # Create approval logic
    approval_conditions = (
        (data['Income'] > 40000) &
        (data['CreditScore'] > 600) &
        (data['LoanAmount'] < data['Income'] * 2) &
        (data['HasDefaulted'] == 0)
    )
    data['Approved'] = np.where(approval_conditions, 1, 0)
    
    return data

df = generate_loan_data()

# User input form
with st.form("loan_form"):
    st.subheader("Applicant Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 20000, 500000, 60000)
        loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000)
    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 700)
        employed_months = st.number_input("Months Employed", 0, 120, 24)
        has_defaulted = st.selectbox("Previous Default?", ["No", "Yes"])
    
    submitted = st.form_submit_button("Predict Approval")

# Preprocessing
def preprocess_input(age, income, loan_amount, credit_score, employed_months, has_defaulted):
    return pd.DataFrame([[
        age,
        income,
        loan_amount,
        credit_score,
        employed_months,
        36,  # Default loan term
        1 if has_defaulted == "Yes" else 0
    ]], columns=df.columns[:-1])

# Model training
@st.cache_resource
def train_model():
    X = df.drop('Approved', axis=1)
    y = df['Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

# Prediction and results
if submitted:
    input_data = preprocess_input(age, income, loan_amount, credit_score, employed_months, has_defaulted)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Approved (Confidence: {proba:.0%})")
        st.balloons()
    else:
        st.error(f"❌ Denied (Confidence: {proba:.0%})")
    
    # Show key factors
    st.write("Key Decision Factors:")
    coef_df = pd.DataFrame({
        'Feature': df.drop('Approved', axis=1).columns,
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False)
    st.dataframe(coef_df)
    
    # Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Approval rate by credit score
    df['CreditScoreBucket'] = pd.cut(df['CreditScore'], bins=5)
    approval_rates = df.groupby('CreditScoreBucket')['Approved'].mean()
    approval_rates.plot(kind='bar', ax=ax[0], color='green')
    ax[0].set_title("Approval Rate by Credit Score")
    ax[0].set_ylabel("Approval Rate")
    
    # Income vs Loan Amount
    ax[1].scatter(
        df[df['Approved']==1]['Income'], 
        df[df['Approved']==1]['LoanAmount'], 
        color='green', label='Approved')
    ax[1].scatter(
        df[df['Approved']==0]['Income'], 
        df[df['Approved']==0]['LoanAmount'], 
        color='red', label='Denied')
    ax[1].set_title("Income vs Loan Amount")
    ax[1].set_xlabel("Income")
    ax[1].set_ylabel("Loan Amount")
    ax[1].legend()
    
    st.pyplot(fig)

# Show raw data option
if st.checkbox("Show sample loan data"):
    st.dataframe(df.head(20))
