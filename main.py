import cv2
from deepface import DeepFace
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

face_classifier = cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("haarcascade_frontalface_default.xml"))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray)
    response = DeepFace.analyze(frame, actions=("emotion",), enforce_detection=False)
    print(response)
    # Initialize new_frame with the original frame
    new_frame = frame.copy()
    
    if response and isinstance(response, list) and len(response) > 0:
        response = response[0]
        for face in faces:  
            x, y, w, h = face
            if "dominant_emotion" in response:
                print(f"Dominant emotion: {response['dominant_emotion']}")
                cv2.putText(new_frame, text=response["dominant_emotion"], org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            new_frame = cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

    cv2.imshow("Emotion Detection", new_frame)
    if cv2.waitKey(30) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()