import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('../face.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face=img[y:y+h,x:x+w]
# Display
    cv2.imshow('img', img)



    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("./data/img.jpg",face)

        break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
