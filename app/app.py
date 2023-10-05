import gradio as gr
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load your pre-trained machine learning model
model = joblib.load(r"C:\Users\IKE\OneDrive - Azubi Africa\Project1\P4-Gradio-Customer-Churn-Machine-Learning-Web-App\model\RFC_model.joblib")

# Load any preprocessing objects used during model training
scaler = joblib.load(r"C:\Users\IKE\OneDrive - Azubi Africa\Project1\P4-Gradio-Customer-Churn-Machine-Learning-Web-App\model\preprocessor.joblib")


# Define a function for making predictions
def predict_churn(
    #customerID,
    gender, SeniorCitizen, Partner, Dependents, tenure, MonthlyCharges
):
    
    
    # Prepare the input data in a DataFrame
    input_data = pd.DataFrame(
        {
            #"customerID": [customerID],
            "gender": [gender],
            "senior_citizen": [SeniorCitizen],
            "partner": [Partner],
            "dependents": [Dependents],
            "tenure": [tenure],
            "MonthlyCharges": [MonthlyCharges],
        }
    )

    # Apply the same preprocessing steps used during training
    input_data = scaler.transform(input_data)


    # Make the churn prediction
    prediction = model.predict(input_data)

    # Return a user-friendly result
    if prediction[0] == 1:
        result = "Churn"
    else:
        result = "No Churn"

    return result


# Create the Gradio interface
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        #gr.components.Textbox(label="Customer ID"),
        gr.components.Dropdown(label="Gender", choices=["Male", "Female", "Other"]),
        gr.components.Number(label="Senior Citizen (0 for No, 1 for Yes)"),
        gr.components.Dropdown(label="Partner", choices=["Yes", "No"]),
        gr.components.Dropdown(label="Dependents", choices=["Yes", "No"]),
        gr.components.Number(label="Tenure (months)"),
        gr.components.Number(label="Monthly Charges"),
    ],
    outputs=gr.components.Textbox(label="Prediction"),
    title="Telco Customer Churn Prediction App",
    description="Please Enter Customer Information to Predict Churn.",
)

# Launch the Gradio interface
interface.launch(debug=True)
