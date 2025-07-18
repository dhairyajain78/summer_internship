# app.py - Deployment Optimized Version
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Critical for deployment
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data  # Cache data for better performance
def load_data():
    np.random.seed(42)
    n_customers = 200
    data = pd.DataFrame({
        'Tenure': np.random.randint(1, 72, n_customers),
        'MonthlyCharges': np.random.uniform(20, 100, n_customers).round(2),
        'TotalCharges': np.random.uniform(50, 5000, n_customers).round(2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
    })
    conditions = [
        (data['Contract'] == 'Month-to-month') & (data['Tenure'] < 12),
        (data['MonthlyCharges'] > 70) & (data['OnlineSecurity'] == 'No')
    ]
    data['Churn'] = np.select(conditions, [1, 1], default=0)
    return data

# Simplified UI setup
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction")
df = load_data()

# Input widgets in columns
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider('Tenure (months)', 1, 72, 12)
    monthly_charges = st.slider('Monthly Charges ($)', 20, 100, 50)
with col2:
    contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    online_security = st.selectbox('Online Security', ('Yes', 'No', 'No internet service'))

# Preprocessing function
@st.cache_data
def preprocess(df):
    df = pd.get_dummies(df, columns=['Contract', 'OnlineSecurity'])
    return df

# Model training
@st.cache_resource  # Cache model for performance
def train_model():
    df_processed = preprocess(df)
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=1000)  # Increased iterations for stability
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

# Prediction
input_data = pd.DataFrame([[
    tenure, monthly_charges, 0,  # Placeholder for TotalCharges
    contract, online_security
]], columns=['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'OnlineSecurity'])

input_processed = preprocess(input_data)
missing_cols = set(preprocess(df).columns) - set(input_processed.columns)
for col in missing_cols:
    if col != 'Churn':
        input_processed[col] = 0
input_processed = input_processed[preprocess(df).drop('Churn', axis=1).columns]

prediction = model.predict(input_processed)[0]
prob = model.predict_proba(input_processed)[0][prediction]

# Display results
st.success(f"Prediction: {'🚨 CHURN' if prediction else '✅ RETAIN'} (Confidence: {prob:.0%})")

# Visualizations
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax[0])
ax[0].set_title('Monthly Charges by Churn Status')

cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title('Confusion Matrix')
st.pyplot(fig)
