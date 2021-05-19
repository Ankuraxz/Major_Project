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

class_list=['n000037', 'n000021', 'n000005', 'n000104', 'n000085', 'n000065', 'n000076', 'n000052', 'n000004', 'n000095', 'n000034', 'n000071', 'n000043', 'n000088', 'n000038', 'n000057', 'n000054', 'n000031', 'n000075', 'n000014', 'n000022', 'n000015', 'n000023', 'n000017', 'n000030', 'n000077', 'n000083', 'n000102', 'Rahul Gupta', 'n000073', 'n000049', 'n000081', 'n000060', 'n000044', 'n000066', 'n000010', 'n000019', 'n000063', 'n000033', 'n000097', 'n000002', 'n000051', 'n000032', 'n000058', 'n000067', 'n000093', 'n000100', 'n000111', 'n000020', 'n000107', 'n000016', 'n000110', 'n000008', 'n000007', 'n000050', 'n000018', 'n000056', 'n000041', 'n000074', 'n000099', 'n000045', 'Mohit Rohilla', 'Animesh ', 'n000012', 'n000101', 'n000048', 'n000013', 'n000105', 'n000035', 'n000061', 'n000047', 'n000092', 'Ritik Saini', 'n000059', 'n000036', 'n000079', 'n000003', 'n000064', 'n000027', 'Ankur', 'n000026', 'n000089', 'n000042', 'n000080', 'n000055', 'n000070', 'n000087', 'n000072', 'n000053', 'n000062', 'n000090', 'n000109', 'n000011', 'n000096', 'n000068', 'n000039', 'n000069', 'n000006', 'n000103', 'n000028', 'n000084', 'n000098', 'n000024', 'n000091', 'n000086', 'n000046', 'n000025', 'n000094', 'n000108']

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

		pred_name =class_list[np.argmax(prediction(imgx))]
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
