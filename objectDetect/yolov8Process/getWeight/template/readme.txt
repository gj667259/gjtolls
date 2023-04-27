1、按照data格式准备数据
2、改data.yaml 、model.yaml
3、运行train.py
4、运行trans.py
5、得到的 onnx权重放到use，改名或者替换yolov8s.onns
6、拿出use文件  detect.py即可识别文件 返回即识别结果