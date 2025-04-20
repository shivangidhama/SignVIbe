import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("sign_language_model.keras")
    print("Model loaded successfully!")
    return model

model = load_model()

# Class labels: A-Z + 1-9
categories = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + [chr(i) for i in range(65, 91)]

# Prediction function
def predict_sign(image_file):
    # Convert to grayscale
    image = Image.open(image_file).convert('L')
    print(f"Original Image Size: {image.size}")  # Debugging: Image size

    # Resize to model input size (64x64)
    image_resized = image.resize((64, 64))
    print(f"Resized Image Size: {image_resized.size}")  # Debugging: Resized image size

    # Normalize and reshape
    img_array = np.array(image_resized).reshape(1, 64, 64, 1) / 255.0
    print(f"Image Shape for Model: {img_array.shape}")  # Debugging: Image shape going to the model

    # Predict
    prediction = model.predict(img_array)
    print(f"Prediction Output: {prediction}")  # Debugging: Raw prediction output
    predicted_class = np.argmax(prediction)
    predicted_label = categories[predicted_class]

    return predicted_label, image_resized

# Streamlit UI
st.title("Sign Language to Text")
st.write("Upload an image of a hand sign (for A-Z or 1-9).")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    img_resized = img.resize((128, 128))  # Resize for display
    st.image(img_resized, caption="Uploaded Image", use_container_width=False)

    if st.button("Predict"):
        label, processed_img = predict_sign(uploaded_file)
        st.success(f"Predicted Sign: **{label}**")
        st.image(processed_img.resize((128, 128)), caption="Processed Input", use_container_width=False)
