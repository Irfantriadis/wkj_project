import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# Set page config
st.set_page_config(page_title="LeafScan Pro", page_icon="🍃", layout="wide")

# Custom CSS to enhance the appearance
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #f0f4f8, #d7e3fc);
    }
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        color: #4CAF50;
        border-radius: 20px;
        border: 2px solid #4CAF50;
        background-color: white;
        padding: 10px 20px;
        font-size: 16px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load the TFLite model and allocate tensors
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Load labels
@st.cache_data
def load_labels():
    with open('labels.txt', 'r') as f:
        return f.read().splitlines()

labels = load_labels()

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input size
    img = np.array(img).astype(np.float32) / 255.0  # Normalize image and convert to FLOAT32
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make predictions
def predict(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
   
    img = preprocess_image(image)
   
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
   
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]
   
    return labels[predicted_index], confidence * 100  # Convert confidence to percentage

# Main Streamlit app
def main():
    st.title("🍃 WKJ")
    st.subheader("Herbal Leaf Detection using CNN")

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Add a spinner while processing
        with st.spinner('Analyzing leaf...'):
            time.sleep(2)  # Simulate processing time
            label, confidence = predict(image)

        with col2:
            st.success("Analysis Complete!")
            st.subheader("Results:")
            st.write(f"**Detected Herb:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Create a progress bar for confidence
            st.progress(confidence / 100)

if __name__ == '__main__':
    main()