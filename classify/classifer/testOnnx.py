import onnxruntime as ort
import numpy as np
import time
import cv2
import re
import yaml

# print(ort.get_device())

# # onnx_model = onnx.load("ImageClassifier.onnx")
# # onnx.checker.check_model(onnx_model) # 检查模型是否符合规范

# model = onnx.load("yolov8s.onnx")
# input_names = [input.name for input in model.graph.input]
# print(input_names)


# 读取图像
img = cv2.imread('./assets/bus.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 缩放到指定大小
resized_img = cv2.resize(img_rgb, (224, 224))

img_np = np.array(resized_img).astype(np.float32)
img_np = img_np.reshape(1, 3, 224, 224)
# print(img_np.shape)
# input_data = np.random.random(size=(1, 3, 640, 640)).astype(np.float32)

resse = ort.InferenceSession("resnet34cpu.onnx",  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print(resse.get_providers(), 'ccccccc')

output = resse.run(None, {"modelInput": img_np})   # "modelInput" 必须和转化时候的参数一样, img_np输入图片形状也必须固定

# print(output2[0][0][0])
