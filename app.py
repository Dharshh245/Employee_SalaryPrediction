import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model
model = pickle.load(open('xgb_model.pkl', 'rb'))

# LabelEncoders (update these based on your training data)
gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
job_encoder = LabelEncoder()

# Set the classes used during model training (replace with actual values)
gender_encoder.classes_ = np.array(['Female', 'Male', 'Other'])  
education_encoder.classes_ = np.array(['Bachelor', 'Master', 'PhD'])  
job_encoder.classes_ = np.array([
    'Software Engineer',
    'Data Scientist',
    'Project Manager',
    'Product Manager',
    'Business Analyst',
    'HR Manager',
    'Sales Executive',
    'Marketing Manager',
    'Financial Analyst',
    'Accountant',
    'Consultant',
    'Network Engineer',
    'System Administrator',
    'UI/UX Designer',
    'Graphic Designer',
    'Mechanical Engineer',
    'Civil Engineer',
    'Electrical Engineer',
    'Teacher',
    'Research Scientist',
    'Doctor',
    'Nurse',
    'Lawyer',
    'Customer Support',
    'Operations Manager'
])

# Streamlit App
st.title("Employee Salary Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", gender_encoder.classes_)
education = st.selectbox("Education Level", education_encoder.classes_)
job = st.selectbox("Job Title", job_encoder.classes_)
experience = st.number_input("Years of Experience", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict Salary"):
    try:
        # Encode categorical fields
        gender_encoded = gender_encoder.transform([gender])[0]
        education_encoded = education_encoder.transform([education])[0]
        job_encoded = job_encoder.transform([job])[0]

        # Create DataFrame in the **exact order of training features**
        input_df = pd.DataFrame([[
            age,
            gender_encoded,
            education_encoded,
            job_encoded,
            experience
        ]], columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Salary: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"Error occurred: {e}")
