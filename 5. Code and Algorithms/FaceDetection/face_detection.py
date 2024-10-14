import cv2

# Enable camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

cap.set(3, 640)  # Set width
cap.set(4, 420)  # Set height

# Import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Uncomment if you want to detect eyes
# eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read from camera.")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor

    # Drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        '''
        # Detecting eyes (if eye detection is needed)
        eyes = eyeCascade.detectMultiScale(imgGray)
        # Drawing bounding box for eyes
        for (ex, ey, ew, eh) in eyes:
            img = cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 3)
        '''

    # Display the resulting frame
    cv2.imshow('face_detect', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
