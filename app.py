import streamlit as st
import cv2
import os
from PIL import Image
import pandas as pd
import tensorflow as tf
from layers import L1Dist
import numpy as np
import datetime
import time
# Import uuid library to generate unique image names
import uuid


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def main():
    f = open("data.csv", "w")
    f.truncate()
    f.close()

    sidebar = st.sidebar.selectbox(
        'Choose one of the following', ('Welcome', 'Face Recognition'))

    # load the model
    model = tf.keras.models.load_model(
        'model.h5', custom_objects={'L1Dist': L1Dist})

    if sidebar == 'Welcome':
        welcome()
    if sidebar == 'Face Recognition':
        face_recognition(model)


def welcome():
    st.title('COS30082 Applied Machine Learning')
    st.header("Group 3: Face recognition attendance system")
    st.image('app_images/welcome.jpg', use_column_width=True)
    st.subheader('Team members:')
    st.text('1. Clement Goh Yung Jing (101218668) \n2. Lee Zhe Sheng (10215371)\n3. Cheryl Tan Shen Wey (101222753) \n4. Vibatha Naveen Jayakody Jayakody Arachcilage (101232163)')


def face_recognition(model):
    st.title("Face Recognition (Web Cam)")

    # Slider for threshold
    # Detection Threshold
    st.subheader('Detection Threshold')
    detection_threshold = st.slider(
        'Try out threshold between 0-1', value=0.50)
    st.write("Detection Threshold:", detection_threshold)

    # Verification Threshold
    st.subheader('Verification Threshold')
    verification_threshold = st.slider(
        'Try out threshold between 0-1', value=0.59)
    st.write("Verification Threshold:", verification_threshold)

    st.caption('Please check the checkbox below for web cam previewing')

    run = st.checkbox('Webcam Preview')

    webcam_preview(run)

    st.subheader('Step 1')

    step_1()

    st.subheader('Step 2')
    step_2()

    st.subheader('Step 3')
    step_3(model, detection_threshold, verification_threshold)


# Load image from file and convert to 105x105 pixel
def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (105, 105))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


def webcam_preview(run):

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)


def step_1():
    register_btn = st.button('Register as new user')

    if register_btn:
        with st.spinner('Registering...'):
            REGISTER_PATH = os.path.join(
                'application_data', 'verification_images')
            camera = cv2.VideoCapture(0)
            REGISTER_PATH2 = os.path.join(
                REGISTER_PATH, '{}.jpg'.format(uuid.uuid1()))
            ret, frame = camera.read()
            cv2.imwrite(REGISTER_PATH2, frame)
        st.success('Register Successfully!')


def step_2():
    capture_btn = st.button("Capture your face")

    if capture_btn:
        with st.spinner('Capturing...'):
            camera = cv2.VideoCapture(0)
            SAVE_PATH = os.path.join(
                'application_data', 'input_image', 'input_image.jpg')
            ret, frame = camera.read()
            cv2.imwrite(SAVE_PATH, frame)
            st.image('application_data/input_image/input_image.jpg',
                     use_column_width=True)
        st.success('Done!')


def step_3(model, detection_threshold, verification_threshold):

    username = st.text_input("Enter you name")
    verify_btn = st.button("Verify")

    if verify_btn:
        with st.spinner('Verifying...'):
            results = []
            for image in os.listdir(os.path.join('application_data', 'verification_images')):
                input_img = preprocess(os.path.join(
                    'application_data', 'input_image', 'input_image.jpg'))
                validation_img = preprocess(os.path.join(
                    'application_data', 'verification_images', image))

                # Make Predictions
                result = model.predict(
                    list(np.expand_dims([input_img, validation_img], axis=1)))
                results.append(result)

            # Detection Threshold: Metric above which a prediciton is considered positive
            detection = np.sum(np.array(results) > detection_threshold)

            # Verification Threshold: Proportion of positive predictions / total positive samples
            verification = detection / \
                len(os.listdir(os.path.join(
                    'application_data', 'verification_images')))
            verified = verification > verification_threshold

            st.text(results)
            st.text(detection)
            st.text(verification)
            st.text(verified)
            st.text('Verified' if verified == True else 'Unverified')

            if verified == True:
                clockin(username)
            else:
                st.text('User is not defined')

        st.success('Done!')


def clockin(username):

    currentDate = datetime.date.today()
    currentTime_in = datetime.datetime.now().time()

    import csv
    with open('data.csv', mode='a') as Clockin:
        Clockin_writer = csv.writer(
            Clockin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        Clockin_writer.writerow([username, currentDate, currentTime_in])


if __name__ == "__main__":
    main()
