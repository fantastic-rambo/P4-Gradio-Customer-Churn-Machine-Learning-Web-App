import gradio as gr
import joblib  # Assuming your RFC model is saved as a joblib file
import numpy as np
from sklearn.preprocessing import StandardScaler  # Import any necessary preprocessing modules

# Load your RFC model
model_path = "path_to_your_rfc_model.joblib"
rfc_model = joblib.load(model_path)

# Load your preprocessing steps, if any
# For example, if you used StandardScaler during training, load it here
scaler = joblib.load("path_to_your_scaler.joblib")

# Define a function for making predictions using the RFC model
def predict(input_data):
    # Preprocess your input data
    # For example, convert input_data to a NumPy array
    input_data = np.array(input_data).reshape(1, -1)

    # Apply any necessary preprocessing steps
    if scaler is not None:
        input_data = scaler.transform(input_data)

    # Make predictions using the RFC model
    prediction = rfc_model.predict(input_data)

    # Return the prediction result
    return prediction[0]

# Create the Gradio interface
input_textbox = gr.inputs.Textbox(label="Input Data")
output_textbox = gr.outputs.Textbox(label="Prediction")

gr.Interface(
    fn=predict,
    inputs=input_textbox,
    outputs=output_textbox,
    layout="vertical",
    title="RFC Model Predictor",
).launch()
