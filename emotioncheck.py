import cv2
from deepface import DeepFace

face_classifier = cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("haarcascade_frontalface_default.xml"))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray)
    response = DeepFace(frame, actoin=("emotions"))
    print(response)
    # Initialize new_frame with the original frame
    new_frame = frame.copy()
    
    for face in faces:  
        x, y, w, h = face
        new_frame = cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

    cv2.imshow("", new_frame)
    if cv2.waitKey(30) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
