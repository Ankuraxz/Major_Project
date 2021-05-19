import onnx
model = onnx.load('./XCEPTION.onnx')
output = model.graph.output
print(output)
