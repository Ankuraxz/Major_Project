import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("./face.xml")
while True:
    	ret,frame = cam.read()
	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(frame,1.3,5)

	if(len(faces)==0): #if no face is seen, the camera hangs up rather than clossing with empty()! error
		cv2.imshow("Faces",frame)
		continue 

	for face in faces:
		x,y,w,h = face

		offset = 10
		face_selection = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection = cv2.resize(face_selection,(100,100))


		#predicted label
		out = knn(trainset,face_selection.flatten())
			#displahy
		pred_name = "Ankur"
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,125),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),5)


	cv2.imshow("Faces",frame)

	key=cv2.waitKey(1) & 0xFF

	if key==ord('q'):
		break

cam.release()
cv2.destroyAllWindows()
© 2021 GitHub, Inc.