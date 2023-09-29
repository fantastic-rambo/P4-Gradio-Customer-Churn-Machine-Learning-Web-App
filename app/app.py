import gradio as gr
from tensorflow import keras

# Load the TensorFlow model
model = keras.models.load_model('my_model.h5')

# Define a function that takes the input data and returns the output of the model
def predict(input_data):
    
    # Apply the model to the input data and return the output
    model_output = model.predict(input_data)
    return model_output

# Create a Gradio interface that takes the input data and displays the output of the model
interface = gr.Interface(fn=predict, inputs="text", outputs="text")

# Launch the Gradio interface
interface.launch()