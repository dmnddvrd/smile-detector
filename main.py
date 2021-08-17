import cv2

face_classifier = cv2.CascadeClassifier('./data/face_detector.xml')
smile_clasisfier = cv2.CascadeClassifier('./data/smile_detector.xml')

# If you get "can't open camera by index" try using -1 instead of 0
webcam = cv2.VideoCapture(0)

# Watching the webcam in a loop
while True:

    # Get a frame from webcam
    frame_read, frame = webcam.read()
    if not frame_read:
        print('Could not read webcam')
        break
    # Changing frame to black and white for optimization
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces/smiles using the classifier loaded
    faces_found = face_classifier.detectMultiScale(frame_grayscale)
    for (x, y, w, h) in faces_found:
        # Drawing a green rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 3)

        # Exracting rectangle with face from photo and grayscale it again
        face = frame[y:y+h, x:x+w]
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Search for smile inside gray rectangle containing the face
        smiles_found = smile_clasisfier.detectMultiScale(
            face_grayscale, scaleFactor=1.1, minNeighbors=20)

        for (x_, y_, w_, h_) in smiles_found:
            # Draw rectangles around smiles inside a faces
            cv2.rectangle(frame, (x+x_, y+y_), (x+x_+w_, y+y_+h_),
                          (100, 200, 250), 1)

        if len(smiles_found):
            cv2.putText(frame, 'Smiling', (x, y+h+40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(100, 200, 250))
    cv2.imshow('Smile detector', frame)
    cv2.waitKey(1)

# Cleanup
cv2.destroyAllWindows()
webcam.release()
