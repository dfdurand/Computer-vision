import streamlit as st
import cv2
import numpy as np


def detect_eyes(image, scaleFactor, minNeighbors):
    # Load the Haar cascade file for eye detection
    eye_cascade = cv2.CascadeClassifier('files/haarcascade_eye.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the grayscale image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw a rectangle around each detected eye
    for (x,y,w,h) in eyes:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    return image, eyes

def detect_faces(image, scaleFactor, minNeighbors):
    # Load the Haar cascade file for face detection
    face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw a rectangle around each detected face
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    return image, faces

def detect_smiles(image, scaleFactor, minNeighbors):
    # Load the Haar cascade file for smile detection
    smile_cascade = cv2.CascadeClassifier('files/haarcascade_smile.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect smiles in the grayscale image
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw a rectangle around each detected smile
    for (x,y,w,h) in smiles:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

    return image , smiles

def read_image(image_file):
    img = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img, 1)
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    # Set the page title and header
    st.set_page_config(page_title='Face and Eye Detection', page_icon=':eyes:')
    st.title('Faces, smiles and Eyes Detection')
    # horizontal alignment
    col1 , col2 = st.columns(2)

    # Upload an image file
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    # Create sliders for the scaleFactor and minNeighbors parameters
    scaleFactor_eye = st.slider('Scale Factor (Eyes)', min_value=1.01, max_value=5.0, step=0.1, value=1.1)
    minNeighbors_eye = st.slider('Min Neighbors (Eyes)', min_value=1, max_value=10, step=1, value=5)
    scaleFactor_face = st.slider('Scale Factor (Faces)', min_value=1.01, max_value=5.0, step=0.1, value=1.2)
    minNeighbors_face = st.slider('Min Neighbors (Faces)', min_value=1, max_value=10, step=1, value=5)
    scaleFactor_smile = st.slider('Scale Factor (Smiles)', min_value=1.01, max_value=5.0, step=0.1, value=1.7)
    minNeighbors_smile = st.slider('Min Neighbors (Smiles)', min_value=1, max_value=10, step=1, value=5)

    if uploaded_file is not None:
        # print("Image Uploaded")
        col1.image(uploaded_file, use_column_width=True)
        out_image = read_image(uploaded_file)
        st.text("Original Image")

        _,faces = detect_faces(out_image, scaleFactor_face, minNeighbors_face)
        _,eyes = detect_eyes(out_image, scaleFactor_eye, minNeighbors_eye)
        _,smiles = detect_smiles(out_image, scaleFactor_smile, minNeighbors_smile)
        
        # st.image(faces, caption='Faces', use_column_width=True)
        # st.image(eyes, caption='Eyes', use_column_width=True)
        # st.image(smiles, caption='Smiles', use_column_width=True)
        col2.image(out_image, use_column_width=True)
        st.image(out_image, caption='Original Image', use_column_width=True)
        st.write("face(s) position: ", faces)
        st.write("eyes position: ", eyes)
        st.write("smile position: ", smiles)




if __name__ == '__main__':
		main()	