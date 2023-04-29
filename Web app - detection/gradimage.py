import cv2
import gradio as gr

face_cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

iface = gr.Interface(fn=detect_faces, inputs="image", outputs="image")

iface.launch()

