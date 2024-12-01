import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = MobileNetV2(weights='imagenet')

def model_predict(img, model):
    """Preprocess the image, make predictions, and return top 3 predictions."""
    img = img.resize((224, 224))  
    x = image.img_to_array(img)  
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x)  
    preds = model.predict(x) 
    return decode_predictions(preds, top=3)[0] 

st.title('Image Classification via MobileNetV2')
st.write("Upload an image and the model will predict the top-3 matching ImageNet classes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    labels = model_predict(image, model)
    for label in labels:
        st.write(f"Prediction (class: {label[1]}, probability: {label[2]*100:.2f}%)")
