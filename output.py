import onnx
model = onnx.load('./INCEPTION_RESNET.onnx')
output = model.graph.output
print(output)
