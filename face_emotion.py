import cv2
from deepface import DeepFace

# Load Haar cascade
face_values = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_values.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

        try:
            analyze = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analyze[0]['dominant_emotion']
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        except:
            print("No face detected")

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()