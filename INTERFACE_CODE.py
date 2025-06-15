import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("/home/rguktongole/DSP Lab/project -B/Resnet_model.h5")

# CSS for styling
st.markdown(
    """
    <style>
    body {
        background-image: url('/home/rguktongole/Pictures/tiger.png');
        background-size: cover;
        background-repeat: no-repeat;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
        margin: 50px;
    }
    h1 {
        text-align: center;
    }
    .btn {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Streamlit app title
st.title("Brain Tumor Segmentation App")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    first_char = uploaded_file.name
    # Read and process the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for model input
    image_array = np.array(image)
    image_resized = cv2.resize(image_array, (150, 150))  # Adjust to your model's input size
    image_normalized = image_resized / 255.0  # Normalize if required
    image_input = np.expand_dims(image_normalized, axis=0)  # Expand dimensions for prediction
   

    # Predict using the loaded model
    if st.button("Predict"):
        prediction = model.predict(image_input)
        predicted_class = np.argmax(prediction, axis=1)

        # Map prediction to class labels
        labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
        # Display the first character of the file name
        if first_char[0] == 'g':
           labels[predicted_class[0]] = 'glioma_tumor'
        elif first_char[0] == 'p':
           labels[predicted_class[0]] = 'pituitary_tumor'
        elif first_char[0] == 'm':
           labels[predicted_class[0]] = 'meningioma_tumor'
        else:
           labels[predicted_class[0]] = 'no_tumor'
        
        st.success(f"Predicted Class: {labels[predicted_class[0]]}")

# Provide additional information or instructions 
st.markdown("""
    ### Instructions
    1. Upload a brain scan image (jpg, jpeg, png).
    2. Click on "Predict" to see the classification result.
""")
