import cv2
import engine as eng
import inference as inf
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER) 

class_list=['n000037', 'n000021', 'n000005', 'n000104', 'n000085', 'n000065', 'n000076', 'n000052', 'n000004', 'n000095', 'n000034', 'n000071', 'n000043', 'n000088', 'n000038', 'n000057', 'n000054', 'n000031', 'n000075', 'n000014', 'n000022', 'n000015', 'n000023', 'n000017', 'n000030', 'n000077', 'n000083', 'n000102', 'Rahul Gupta', 'n000073', 'n000049', 'n000081', 'n000060', 'n000044', 'n000066', 'n000010', 'n000019', 'n000063', 'n000033', 'n000097', 'n000002', 'n000051', 'n000032', 'n000058', 'n000067', 'n000093', 'n000100', 'n000111', 'n000020', 'n000107', 'n000016', 'n000110', 'n000008', 'n000007', 'n000050', 'n000018', 'n000056', 'n000041', 'n000074', 'n000099', 'n000045', 'Mohit Rohilla', 'Animesh ', 'n000012', 'n000101', 'n000048', 'n000013', 'n000105', 'n000035', 'n000061', 'n000047', 'n000092', 'Ritik Saini', 'n000059', 'n000036', 'n000079', 'n000003', 'n000064', 'n000027', 'Ankur', 'n000026', 'n000089', 'n000042', 'n000080', 'n000055', 'n000070', 'n000087', 'n000072', 'n000053', 'n000062', 'n000090', 'n000109', 'n000011', 'n000096', 'n000068', 'n000039', 'n000069', 'n000006', 'n000103', 'n000028', 'n000084', 'n000098', 'n000024', 'n000091', 'n000086', 'n000046', 'n000025', 'n000094', 'n000108']

CLASSES =109
HEIGHT = 224
WIDTH = 224
#SHAPE =[1,224,224,3] #RGB IMAGE

x1 = "./data/def_img.jpg"
x2 = "./data/uni_img.jpg"
x3 = "./data/var_img.jpg"
x4 = "./data/ror_img.jpg"

onnx_file = "./lbp_irv2.onnx"
serialized_plan_fp32 = "./lbp.plan"

image1 = cv2.imread(x1,cv2.IMREAD_GRAYSCALE)
#image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
img1 = cv2.resize(image1, (224,224),interpolation = cv2.INTER_NEAREST)

image2 = cv2.imread(x2,cv2.IMREAD_GRAYSCALE)
#image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(image2, (224,224),interpolation = cv2.INTER_NEAREST)

image3 = cv2.imread(x3,cv2.IMREAD_GRAYSCALE)
#image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
img3 = cv2.resize(image3, (224,224),interpolation = cv2.INTER_NEAREST)

image4 = cv2.imread(x4,cv2.IMREAD_GRAYSCALE)
#image4 = cv2.cvtColor(image4,cv2.COLOR_BGR2GRAY)
img4 = cv2.resize(image4, (224,224),interpolation = cv2.INTER_NEAREST)

engine = eng.load_engine(trt_runtime, serialized_plan_fp32)

# PREDICTION ON DEFAULT
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
out1 = inf.do_inference(engine, img1, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
out2 = inf.do_inference(engine,img2,h_input,d_input,h_output,d_output,stream,1,HEIGHT,WIDTH)
out3 = inf.do_inference(engine, img3, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
out4 = inf.do_inference(engine,img4,h_input,d_input,h_output,d_output,stream,1,HEIGHT,WIDTH)

print("OUTPUT OF DEF is",class_list[np.argmax(out1)])
print("OUTPUT OF UNI is",class_list[np.argmax(out2)])
print("OUTPUT OF VAR is",class_list[np.argmax(out3)])
print("OUTPUT OF ROR is",class_list[np.argmax(out4)])

out = (out1+out2+out3+out4)/4
print(out)

print(class_list[np.argmax(out)])


