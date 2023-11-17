# 88922025_Churning_Customer
# Churn Prediction Web App

## Overview
This web app is designed to predict customer churn in a subscription-based business. It utilizes a machine learning model to analyze user input and predict whether a customer is likely to churn or not.

## Features
- **User-Friendly Interface:** The web app provides an intuitive interface for users to input relevant features for predicting churn.
- **Real-time Prediction:** Users can receive real-time predictions on whether a customer is likely to churn based on their input.
- **Confidence Level:** The app displays the confidence level of the prediction, helping users understand the model's certainty.
- **Feature Explanations:** Users can find explanations for the features used in the prediction in the sidebar.

## Technologies Used
- Streamlit: Used for creating the interactive web app.
- TensorFlow and Keras: Utilized for building and training the machine learning model.
- Scikit-learn: Employed for data preprocessing and scaling.

## Files
- `App.py`: Python script containing the deployment code for the Streamlit app.
- `88922025_Churning_Customers.ipynb`: Python script containing the code for training the Churn Prediction model.
- `finalmodel.h5`: The trained machine learning model saved in HDF5 format.
- `scaler.pkl`: The scaler used for preprocessing data saved using Pickle.

## Instructions for Running Locally
1. Install the required Python libraries by running: `pip install -r requirements.txt`.
2. Run the Streamlit app by executing: `streamlit run App.py`.
3. Access the web app in your browser at the provided URL.

## How to Use
1. Input customer details in the sidebar, such as Monthly Charges, Total Charges, Tenure, etc.
2. Click the "Predict" button to obtain the churn prediction and confidence level.
3. Explore feature explanations in the sidebar.
4. Optionally, use the "Clear" button to reset the input.

## Future Enhancements
- Integration with a database for storing and retrieving customer data.
- Inclusion of more advanced visualization of churn trends.
- Enhancement of the model with additional features for improved accuracy.

#Video deployment Code
-https://drive.google.com/file/d/1HMxOQCFHpXfVMJToN6kBATgIE0sNLofV/view?usp=drive_link
Feel free to contribute or provide feedback to improve this project!

