import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

# Title
st.title(" Facial Sentiment Analysis Web App")
st.write("Upload an image to detect emotions using DeepFace.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Display original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Analyze emotions
    with st.spinner("Analyzing emotions..."):
        try:
            result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)

            # Show results
            st.subheader(" Detected Emotions:")
            dominant_emotion = result[0]['dominant_emotion']
            st.write(f"**Dominant Emotion:** {dominant_emotion}")
            st.json(result[0]['emotion'])

            # Draw bounding box around face(s)
            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_array, face['dominant_emotion'], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            st.image(img_array, caption="Processed Image with Detected Face(s)", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
