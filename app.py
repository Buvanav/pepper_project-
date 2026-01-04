import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your model
model = load_model('pepper_mobilenet_model.h5')

st.title("Pepper Health Detector 🌶️")

# Upload image
uploaded_file = st.file_uploader("Upload a pepper image", type=["jpg","png"])
if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred = model.predict(img_array)[0][0]
    
    # Show result
    if pred > 0.5:
        st.success("Unhealthy Pepper 🌶️")
    else:
        st.success("Healthy Pepper ✅")
