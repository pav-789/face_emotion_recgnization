import cv2
from deepface import DeepFace
from datetime import datetime
import csv

# Load face cascade
face_values = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
video = cv2.VideoCapture(0)

# Create/Open a CSV file
with open("emotion_log.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Emotion"])  # Header only once

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_values.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            try:
                analyze = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = analyze[0]['dominant_emotion']
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([timestamp, emotion])  # Save to CSV
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            except:
                print("No face detected")

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()