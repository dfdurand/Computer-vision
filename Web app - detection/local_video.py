import cv2

# Load the Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('files/haarcascade_eye.xml')

# Open the video file
#cap = cv2.VideoCapture('datas/waterb.mp4')
cap = cv2.VideoCapture(0)
# Loop through the frames of the video
while cap.isOpened():
    # Read a frame from the video file
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Draw a rectangle around each detected face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # Detect eyes in the region of interest (ROI) which is the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the frame with detected faces and eyes
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
