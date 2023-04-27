from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')
model.train(data='ftdata.yaml', epochs=300, device='0,1,2 or cpu')