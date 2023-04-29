import cv2
import numpy as np
import gradio as gr

face_cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def detect_faces_video(input_video):
    cap = cv2.VideoCapture(input_video)
    output_video = 'output1.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(cap.get(3)),int(cap.get(4))))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            output_frame = detect_faces(frame)
            output_frame =  cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            out.write(output_frame)
        else:
            break
    cap.release()
    out.release()
    return output_video

iface = gr.Interface(detect_faces_video, 
                     inputs="video", 
                     outputs="file", 
                     capture_session=True, 
                     title="Face Detection in Video",
                     description="Upload a video to detect faces in it and generate an output video.")
iface.launch()


