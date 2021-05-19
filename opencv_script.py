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

class_dict={'Animesh ': 0, 'Ankur': 1, 'Mohit Rohilla': 2, 'Rahul Gupta': 3, 'Ritik Saini': 4, 'n000002': 5, 'n000003': 6, 'n000004': 7, 'n000005': 8, 'n000006': 9, 'n000007': 10, 'n000008': 11, 'n000010': 12, 'n000011': 13, 'n000012': 14, 'n000013': 15, 'n000014': 16, 'n000015': 17, 'n000016': 18, 'n000017': 19, 'n000018': 20, 'n000019': 21, 'n000020': 22, 'n000021': 23, 'n000022': 24, 'n000023': 25, 'n000024': 26, 'n000025': 27, 'n000026': 28, 'n000027': 29, 'n000028': 30, 'n000030': 31, 'n000031': 32, 'n000032': 33, 'n000033': 34, 'n000034': 35, 'n000035': 36, 'n000036': 37, 'n000037': 38, 'n000038': 39, 'n000039': 40, 'n000041': 41, 'n000042': 42, 'n000043': 43, 'n000044': 44, 'n000045': 45, 'n000046': 46, 'n000047': 47, 'n000048': 48, 'n000049': 49, 'n000050': 50, 'n000051': 51, 'n000052': 52, 'n000053': 53, 'n000054': 54, 'n000055': 55, 'n000056': 56, 'n000057': 57, 'n000058': 58, 'n000059': 59, 'n000060': 60, 'n000061': 61, 'n000062': 62, 'n000063': 63, 'n000064': 64, 'n000065': 65, 'n000066': 66, 'n000067': 67, 'n000068': 68, 'n000069': 69, 'n000070': 70, 'n000071': 71, 'n000072': 72, 'n000073': 73, 'n000074': 74, 'n000075': 75, 'n000076': 76, 'n000077': 77, 'n000079': 78, 'n000080': 79, 'n000081': 80, 'n000083': 81, 'n000084': 82, 'n000085': 83, 'n000086': 84, 'n000087': 85, 'n000088': 86, 'n000089': 87, 'n000090': 88, 'n000091': 89, 'n000092': 90, 'n000093': 91, 'n000094': 92, 'n000095': 93, 'n000096': 94, 'n000097': 95, 'n000098': 96, 'n000099': 97, 'n000100': 98, 'n000101': 99, 'n000102': 100, 'n000103': 101, 'n000104': 102, 'n000105': 103, 'n000107': 104, 'n000108': 105, 'n000109': 106, 'n000110': 107, 'n000111': 108}

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

		pred_name =str(class_dict.get(np.argmax(prediction(imgx))))
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
