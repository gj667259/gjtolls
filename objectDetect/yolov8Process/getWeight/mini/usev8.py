from ultralytics import YOLO
import torch

# 参考官方文档

# # Load a model
# model = YOLO('tire8n.yaml')  # build a new model from YAML
# model = YOLO('./runs/detect/train/weights/best.pt')  # load a pretrained model (recommended for training)
# model = YOLO('tire8n.yaml').load('./runs/detect/train/weights/best.pt')  # build from YAML and transfer weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Train the model
# model.train(data='tiredata.yaml', epochs=100, imgsz=640, batch=2, workers=4, device='0')
# res = model('./418j/Image_20230418163907241.jpg')


# Use the model
m = YOLO("yolov8s.pt").val()  # load a pretrained model (recommended for training)
# m.to(device)
results = m("bus.jpg")  # predict on an image
print(results)


# while True:
#     results = model("1.jpg")  # predict on an image

# success = m.export(format="TorchScript", device=0)  # export the model to ONNX format
# success = m.export(format="onnx")  # export the model to ONNX format
# print(success)
