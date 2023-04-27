from ultralytics import YOLO

model = YOLO('/runs/detect/train/weights/best.pt')

model.export(format='onnx')
