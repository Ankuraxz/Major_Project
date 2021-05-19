#!/usr/bin/python
__author__      = "Ankur"
__copyright__   = "Copyright 2021 GitHub,Inc.  @Ankuraxz"


import cv2
import numpy as np
import os
import engine as eng
import inference as inf
import tensorrt as trt

# PARAMETERS
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER) 

class_list=[]

CLASSES =109
HEIGHT = 224
WIDTH = 224
SHAPE =[1,224,224,3]

nnx_file = "./Xception.onnx"
serialized_plan_fp32 = "./Xception.plan"
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
engine = eng.load_engine(trt_runtime, serialized_plan_fp32)

#PREDICITON FUNCTION
def prediction( img1):
    	out = inf.do_inference(engine, img1, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
		return out

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
		imgx = cv2.cvtColor(face_selection,cv2.COLOR_RGB2GRAY)

		#predicted label
		
			#displahy
		pred_name = prediction(imgx)
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,125),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),5)


	cv2.imshow("Faces",frame)

	key=cv2.waitKey(1) & 0xFF

	if key==ord('q'):
    		break

cam.release()
cv2.destroyAllWindows()
# © 2021 GitHub, Inc.
# © Author @Ankuraxz