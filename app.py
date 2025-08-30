import streamlit as st
import cv2
from fer import FER
from PIL import Image
import numpy as np

st.title("😊 Real-Time Facial Sentiment Analysis")
st.write("Upload an image and the app will detect emotions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Run FER detector
    detector = FER(mtcnn=True)
    emotions = detector.detect_emotions(img_array)

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if emotions:
        st.subheader("Detected Emotions")
        for face in emotions:
            st.write(face["emotions"])
    else:
        st.write("No face detected. Try another image.")

          
