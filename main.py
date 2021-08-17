import cv2

face_classifier = cv2.CascadeClassifier('./data/face_detector.xml')

# If you get "can't open camera by index" try using -1 instead of 0
webcam = cv2.VideoCapture(0)

# Watching the webcam in a loop
while True:

    # Get a frame from webcam
    frame_read, frame = webcam.read()
    if not frame_read:
        break

    # Changing frame to black and white for optimization
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces using the classifier loaded
    faces_found = face_classifier.detectMultiScale(frame_grayscale)
    if len(faces_found):
        print('Faces found', len(faces_found))
        # Drawing rectangle around face if found
        for (x, y, w, h) in faces_found:
            print(x, y, x+w, y+h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 1)

    cv2.imshow('Smile detector', frame)
    cv2.waitKey(1)

# Cleanup
webcam.release()
cv2.destroyAllWindows()
