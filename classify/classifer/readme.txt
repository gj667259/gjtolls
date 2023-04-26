dLoader.py  ：加载自定义数据集

getAugument.py ： parser获取参数示例

mcTrain.py ： 多个模型可选择，可以终端参数训练

simpleTrain.py ：简单训练示例，模板

train.py ：有tqdm动态训练代码

toOnnx.py ： 将模型转为ONNX版本

testOnnx.py ：实现得到的ONNX版本模型推理

weight : 存放模型权重、预训练、以训练、ONNX

model ： 存放模型,可以通过 getM 问价导入加载各自模型

data ：存放数据，不同加载方式，数据格式也不同