import pickle

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Load the trained model
best_model_path = "finalmodel.h5"
best_model = keras.models.load_model(best_model_path)

# Load the scaler
scaler_path = "scaler.pkl"
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the web app
st.title("Churn Prediction Web App")

# Sidebar for user input features
st.sidebar.header("User Input Features")

monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=500.0, step=5.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=8684.8, step=10.0)
tenure = st.number_input('Tenure', min_value=0.0, max_value=100.0, step=1.0)
gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Partner', ['No', 'Yes'])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two years"])
paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
internet_service_fiber_optic = st.selectbox("Internet Service (Fiber Optic)", ["No", "Yes"])
online_security_no = st.selectbox("Online Security", ["No", "Yes"])
tech_support_no = st.selectbox("Tech Support", ["No", "Yes"])
payment_method_electronic_check = st.selectbox("Payment Method (Electronic Check)", ["No", "Yes"])
dependents = st.selectbox('Dependents', ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ['No', 'Yes'])

# Button to trigger the prediction
predict_button = st.button("Predict")
clear_button = st.button("Clear")

# Explanation in the sidebar
st.sidebar.subheader('Features used for Prediction')
st.sidebar.write('- **Total Charges:** The total amount of charges incurred by the customer over the past year.')
st.sidebar.write('- **Monthly Charges:** The customer\'s monthly bill amount.')
st.sidebar.write('- **Tenure:** The number of months the customer has been a customer.')
st.sidebar.write('- **Contract:** The type of contract the customer has with the company (month-to-month, one year, or two years).')
st.sidebar.write('- **Internet Service:** The type of internet service the customer has (DSL, fiber optic, or no internet).')
st.sidebar.write('- **Payment Method:** The method the customer uses to pay their bill (bank transfer, credit card, electronic check, or mailed check).')
st.sidebar.write('- **Online Security:** Whether or not the customer has online security enabled on their account.')
st.sidebar.write('- **Gender:** The customer\'s gender.')
st.sidebar.write('- **Partner:** Whether or not the customer has a partner.')
st.sidebar.write('- **Senior Citizen:** Whether or not the customer is a senior citizen.')
st.sidebar.write('- **Dependents:** Whether or not the customer has dependents.')
st.sidebar.write('- **Paperless Billing:** Whether or not the customer has paperless billing enabled on their account.')
st.sidebar.write('- **Tech Support:** Whether or not the customer has tech support enabled on their account.')
st.sidebar.write('- **Online Backup:** Whether or not the customer has online backup enabled on their account.')

if predict_button:
    # Convert categorical features to numerical values
    gender = 1 if gender == "Male" else 0
    senior_citizen = 1 if senior_citizen == "Yes" else 0
    partner = 1 if partner == "Yes" else 0
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two years": 2}
    contract = contract_mapping[contract]
    paperless_billing = 1 if paperless_billing == "Yes" else 0
    internet_service_fiber_optic = 1 if internet_service_fiber_optic == "Yes" else 0
    online_security_no = 1 if online_security_no == "Yes" else 0
    tech_support_no = 1 if tech_support_no == "Yes" else 0
    payment_method_electronic_check = 1 if payment_method_electronic_check == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0
    online_backup = 1 if online_backup == "Yes" else 0



    top_features = ['TotalCharges', 'MonthlyCharges', 'tenure', 'InternetService_Fiber optic',
                'Contract_Two year', 'PaymentMethod_Electronic check', 'gender', 'PaperlessBilling',
                'Partner', 'OnlineSecurity', 'Contract_One year', 'TechSupport', 'OnlineBackup',
                'SeniorCitizen', 'Dependents']

# ... (rest of your code)

if predict_button:
    # ... (other code)

    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "tenure": [tenure],
        "PaperlessBilling": [paperless_billing],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Dependents": [dependents],
    })

    # Initialize missing columns to 0
    missing_columns = set(top_features) - set(user_input.columns)
    for column in missing_columns:
        user_input[column] = [0]

    # Reorder columns to match top_features order
    user_input = user_input[top_features]

    # Assuming that your scaler was fit only on numerical features
    user_input_scaled = scaler.transform(user_input)

    # Make the prediction
    prediction = best_model.predict(user_input_scaled)

    confidence_rate = round(max(prediction[0, 0], 1 - prediction[0, 0]) * 100, 2)
    st.write(f"The Prediction is {confidence_rate}% confident.")

    if prediction[0] > 0.3:
        st.write("Customer is likely to churn")
    else:
        st.write("Customer is not likely to churn")


if clear_button:
    st.experimental_rerun()
