import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="📉")

# Title
st.title("📉 Customer Churn Prediction")
st.write("""
Predict if a customer will stop using your service based on their behavior.
""")

# Sample data generation
def generate_churn_data():
    np.random.seed(42)
    n_customers = 200
    
    data = pd.DataFrame({
        'Tenure': np.random.randint(1, 72, n_customers),  # months
        'MonthlyCharges': np.random.uniform(20, 100, n_customers).round(2),
        'TotalCharges': np.random.uniform(50, 5000, n_customers).round(2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
    })
    
    # Create target (more likely to churn with month-to-month contracts)
    conditions = [
        (data['Contract'] == 'Month-to-month') & (data['Tenure'] < 12),
        (data['MonthlyCharges'] > 70) & (data['OnlineSecurity'] == 'No')
    ]
    choices = [1, 1]  # 1 = Churn
    data['Churn'] = np.select(conditions, choices, default=0)
    
    return data

df = generate_churn_data()

# Sidebar for user input
st.sidebar.header("Customer Details")

def user_input_features():
    tenure = st.sidebar.slider('Tenure (months)', 1, 72, 12)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 20, 100, 50)
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    online_security = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
    
    data = {
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'OnlineSecurity': online_security
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user input
st.subheader('Customer Input')
st.write(input_df)

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    # Convert categorical to dummy variables
    df = pd.get_dummies(df, columns=['Contract', 'OnlineSecurity'])
    return df

df_processed = preprocess_data(df)
input_processed = preprocess_data(input_df)

# Ensure all columns exist
for col in df_processed.drop('Churn', axis=1).columns:
    if col not in input_processed:
        input_processed[col] = 0

# Reorder columns
input_processed = input_processed[df_processed.drop('Churn', axis=1).columns]

# Train model
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(input_processed)
prediction_proba = model.predict_proba(input_processed)

st.subheader('Prediction')
churn_status = "🚨 Likely to Churn" if prediction[0] == 1 else "✅ Likely to Stay"
st.write(f"Prediction: **{churn_status}**")
st.write(f"Confidence: {prediction_proba[0][prediction[0]]:.2%}")

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader('Model Performance')
st.write(f"Accuracy: {accuracy:.2%}")

# Confusion matrix
st.write("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Show sample data
if st.checkbox('Show Raw Data'):
    st.subheader('Sample Customer Data')
    st.write(df)

# Key insights
st.subheader('Key Insights')
st.write("""
- Customers with **month-to-month contracts** are more likely to churn
- Higher **monthly charges** increase churn risk
- Having **online security** reduces churn probability
""")
