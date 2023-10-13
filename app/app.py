# Import Libraries
import joblib
import pandas as pd
import numpy as np
import gradio as gr

# Load the preprocessor, the imputer and the Random Forest Classifier model
# Specify the path to the model directory.
model_dir = "src/model"

# Load the preprocessor.
preprocessor = joblib.load(r"C:\Users\IKE\OneDrive - Azubi Africa\Project1\P4-ChurnPredict-Pro\src\model\preprocessor.joblib")

# Load the model
model = joblib.load(r"C:\Users\IKE\OneDrive - Azubi Africa\Project1\P4-ChurnPredict-Pro\src\model\rf_model.joblib")


expected_inputs = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def churn_predict(*args):
    # Convert the tuple to a DataFrame
    df = pd.DataFrame([args], columns=expected_inputs)

    # Convert SeniorCitizen from Yes/No to 1/0
    df["SeniorCitizen"] = df["SeniorCitizen"].replace(["Yes", "No"], ["1", "0"])

    # Apply the preprocessor to transform the use input
    preprocessed_df = preprocessor.transform(df)

    # Predict using the best Random Forest Classifier Model
    pred = model.predict(preprocessed_df)

    # Return both the prediction and user information
    return df, {
        "Prediction: CHURN": float(pred[0]),
        "Prediction: STAY": 1 - float(pred[0]),
    }


# Define some variable limits and lists of options
max_tenure = (
    1.61803398875 * 72
)  # Applied the Golden Ratio to the maximum value from the training data to leave room for increased customer tenures while still ensuring a limit on the possible inputs.
max_monthly_charges = (
    1.61803398875 * 200
)  # Applied the Golden Ratio to the maximum amount of monthly charges from the training data to leave room for increased amounts while still ensuring a limit on the possible inputs.
max_total_charges = (
    1.61803398875 * 8684.8
)  # Applied the Golden Ratio to the maximum amount of total charges from the training data to leave room for increased amounts while still ensuring a limit on the possible inputs.
yes_or_no = [
    "Yes",
    "No",
]  # To be used for the variables whose possible options are "Yes" or "No".
internet_service_choices = [
    "Yes",
    "No",
    "No internet service",
]  # A variable for the choices available for the "Internet Service" variable


# App
with gr.Blocks() as app:
    # Title
    gr.Markdown("# ChurnPredict Pro")

    # About app and Data dictionary
    with gr.Row():
        # Expander for more info on columns
        with gr.Column():
            with gr.Accordion("ABOUT APP", open=False):
                gr.Markdown(
                    """#### ChurnPredict Pro, powered by a trained Random Forest Classifier on customer information, is your go-to tool for accurate churn prediction. Leverage customer data to forecast churn, enhance retention strategies, and maximize customer satisfaction. Unlock the power of data-driven decision-making today!
                    """
                )
            # Expander for more info on columns
        with gr.Column():
            with gr.Accordion("DATA DICTIONARY", open=False):
                gr.Markdown(
                    """This app receives the following as inputs and processes them to return the prediction on whether a customer, given the inputs, will churn or not.
               
                - Gender          : Whether the customer is a male or a female                          
                - SeniorCitizen   : Whether a customer is a senior citizen or not                       
                - Partner         :  Whether the customer has a partner or not (Yes, No)                 
                - Dependents      :  Whether the customer has dependents or not (Yes, No)                
                - Tenure          :  Number of months the customer has stayed with the company           
                - Phone Service   :  Whether the customer has a phone service or not (Yes, No)           
                - MultipleLines   :  Whether the customer has multiple lines or not                      
                - InternetService :  Customer's internet service provider (DSL, Fiber Optic, No)        
                - OnlineSecurity  :  Whether the customer has online security or not (Yes, No, No Internet) 
                - OnlineBackup    :  Whether the customer has online backup or not (Yes, No, No Internet) 
                - DeviceProtection:  Whether the customer has device protection or not (Yes, No, No internet service) 
                - TechSupport     :  Whether the customer has tech support or not (Yes, No, No internet) 
                - StreamingTV     :  Whether the customer has streaming TV or not (Yes, No, No internet service) 
                - StreamingMovies :  Whether the customer has streaming movies or not (Yes, No, No Internet service) 
                - Contract        :  The contract term of the customer (Month-to-Month, One year, Two year) 
                - PaperlessBilling:  Whether the customer has paperless billing or not (Yes, No)        
                - Payment Method  :  The customer's payment method (Electronic check, mailed check, Bank transfer(automatic), Credit card(automatic)) 
                - MonthlyCharges  :  The amount charged to the customer monthly                         
                - TotalCharges    :  The total amount charged to the customer                            
                """
                )

    #  Customer  Information
    gr.Markdown("**CUSTOMER DEMOGRAPHIC INFORMATION**")
    with gr.Row():
        gender = gr.Dropdown(label="Gender", choices=["Female", "Male"], value="Female")
        SeniorCitizen = gr.Radio(label="Senior Citizen", choices=yes_or_no, value="No")
        Partner = gr.Radio(label="Partner", choices=yes_or_no, value="No")
        Dependents = gr.Radio(label="Dependents", choices=yes_or_no, value="No")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Contract and Tenure Data**")
            Contract = gr.Dropdown(
                label="Contract",
                choices=["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
            )
            tenure = gr.Slider(
                label="Tenure (months)",
                minimum=1,
                step=1,
                interactive=True,
                value=1,
                maximum=max_tenure,
            )

        # CUSTOMER PHONE SERVICE USAGE INFORMATION
        with gr.Column():
            gr.Markdown("**CUSTOMER PHONE SERVICE USAGE INFORMATION**")
            PhoneService = gr.Radio(
                label="Phone Service", choices=yes_or_no, value="Yes"
            )
            MultipleLines = gr.Dropdown(
                label="Multiple Lines",
                choices=["Yes", "No", "No phone service"],
                value="No",
            )

    # CUSTOMER INTERNET SERVICE USAGE INFORMATION
    gr.Markdown("**CUSTOMER INTERNET SERVICE USAGE INFORMATION**")
    with gr.Row():
        InternetService = gr.Dropdown(
            label="Internet Service",
            choices=["DSL", "Fiber optic", "No"],
            value="Fiber optic",
        )
        OnlineSecurity = gr.Dropdown(
            label="Online Security", choices=internet_service_choices, value="No"
        )
        OnlineBackup = gr.Dropdown(
            label="Online Backup", choices=internet_service_choices, value="No"
        )
        DeviceProtection = gr.Dropdown(
            label="Device Protection", choices=internet_service_choices, value="No"
        )
        TechSupport = gr.Dropdown(
            label="Tech Support", choices=internet_service_choices, value="No"
        )
        StreamingTV = gr.Dropdown(
            label="TV Streaming", choices=internet_service_choices, value="No"
        )
        StreamingMovies = gr.Dropdown(
            label="Movie Streaming", choices=internet_service_choices, value="No"
        )

    # CUSTOMER BILLING AND PAYMENT INFORMATION
    gr.Markdown("**CUSTOMER BILLING AND PAYMENT INFORMATION**")
    with gr.Row():
        MonthlyCharges = gr.Slider(
            label="Monthly Charges", step=0.05, maximum=max_monthly_charges
        )
        TotalCharges = gr.Slider(
            label="Total Charges", step=0.05, maximum=max_total_charges
        )
        PaperlessBilling = gr.Radio(
            label="Paperless Billing", choices=yes_or_no, value="Yes"
        )
        PaymentMethod = gr.Dropdown(
            label="Payment Method",
            choices=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            value="Electronic check",
        )

    # Submit to make a prediction
    submit_button = gr.Button("Submit")

    # Output Prediction
    output = [
        gr.Dataframe(label="Customer Information", headers=expected_inputs),
        gr.Label(label="Prediction", value="Awaiting Submission..."),
    ]

    submit_button.click(
        fn=churn_predict,
        outputs=output,
        inputs=[
            gender,
            SeniorCitizen,
            Partner,
            Dependents,
            tenure,
            PhoneService,
            MultipleLines,
            InternetService,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            Contract,
            PaperlessBilling,
            PaymentMethod,
            MonthlyCharges,
            TotalCharges,
        ],
    )


app.launch(share=True, debug=True)
