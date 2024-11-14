import os
import tempfile
import cv2
import streamlit as st
from helper import detect_plate
from PIL import Image

# Title
st.title("Plate Recognition System 🚗")
st.write("This app will help you to detect vehicle number plates from uploaded images or videos.")

# Header
st.header("Upload an Image or a Video ")

# File Upload (both image and video)
file = st.file_uploader("", type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"])

# Model Path
model_path = "models/plate_detection.pt"

# If file is uploaded
if file is not None:
    # Check if the uploaded file is an image or video
    if file.type in ["image/png", "image/jpeg", "image/jpg"]:
        # Handle Image Upload
        st.header("Original Image")
        image = Image.open(file).convert('RGB')
        st.image(image, use_container_width=True)

        # Detect plate in the image
        detection_result, cropped_image, is_detected = detect_plate(image, model_path)

        if is_detected != 0:
            st.write("#### [INFO].. Plate is detected!")
            st.image(detection_result, use_container_width=True)
            st.image(cropped_image, use_container_width=True)
        else:
            st.write("#### [INFO].. No plate detected!")
            st.image(detection_result, use_container_width=True)
            st.image(cropped_image, use_container_width=True)

    elif file.type in ["video/mp4", "video/avi", "video/mov", "video/mkv"]:
        # Handle Video Upload
        # Save the uploaded video file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        # Open the video file
        video_stream = cv2.VideoCapture(temp_file_path)

        if not video_stream.isOpened():
            st.error("[ERROR] Failed to open video.")
        else:
            stframe = st.empty()  # Create a placeholder for displaying frames

            while True:
                ret, frame = video_stream.read()

                if not ret:
                    break  # End of video

                # Detect plates in the current frame
                detection_result, cropped_image, is_detected = detect_plate(frame, model_path)

                # Display the detection result in Streamlit
                stframe.image(detection_result, channels="BGR", use_container_width=True)

            # Close the video stream when done
            video_stream.release()
            os.remove(temp_file_path)  # Clean up the temporary video file