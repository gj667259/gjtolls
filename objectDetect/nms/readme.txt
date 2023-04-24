yolo v8 转 onnx之后，导入torch会增加显存，为节约显存，nms使用numpy版本

文件说明
result.pt  一张yolov8识别出的boxes的结果， 结构 （batch， 84， 8400） tensor
    84 = x, y, w, h + classConf
    8400 = 80*80 + 40*40 +20*20

nms_numpy_v8.py  numpy版本nms针对yolov8,因为yolov8没有box conf, 所以使用class最大的当作box conf, 内部做了处理

nms_torch_v8.py     torch版本nms,