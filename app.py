import streamlit as st
import cv2
from fer import FER
from PIL import Image
import numpy as np

st.title("Facial Sentiment Analysis")

img_file = st.camera_input("Take a picture")

if img_file:
    # Convert uploaded file to OpenCV image
    img = Image.open(img_file)
    img_array = np.array(img)

    detector = FER(mtcnn=True)
    result = detector.detect_emotions(img_array)

    st.image(img_array, caption="Your photo", use_column_width=True)

    if result:
        emotions = result[0]["emotions"]
        st.write("Detected emotions:", emotions)
    else:
        st.write("No face detected 😢")
