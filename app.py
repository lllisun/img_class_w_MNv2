import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def predict_image(model, img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return [(class_name, float(prob) * 100) for (number, class_name, prob) in decoded_predictions]

def main():
    st.set_page_config(page_title='MobileNetV2 Image Identifier', page_icon=':camera:')
    st.title('Image Identification with MobileNetV2')
    st.write('Upload an image and get top 3 predictions using MobileNetV2')
    
    model = load_model()
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image to identify its contents"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with st.spinner('Identifying image contents...'):
            predictions = predict_image(model, image)
        
        st.subheader('Top 3 Predictions:')
        for i, (class_name, probability) in enumerate(predictions, 1):
            st.write(f"{i}. {class_name.replace('_', ' ').title()} - {probability:.2f}%")
    
    st.sidebar.header('About')
    st.sidebar.info(
        "This app uses MobileNetV2, a lightweight convolutional neural network "
        "pre-trained on the ImageNet dataset. It can identify 1000 different "
        "object categories with high accuracy."
    )

