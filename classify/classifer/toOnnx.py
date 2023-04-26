import torch.onnx 
from model import resnet, alex


#Function to Convert to ONNX 
def Convert_ONNX(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的AlexNet模型
    # model =  EfficientNet.from_pretrained(model_name='efficientnet' ,weights_path='efficientnet-b0')
    model = resnet.resnet34(num_classes = 2)


    model.load_state_dict(torch.load('./data/level0/resnet34.pth'))
    # model.to(device) 可以不用

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, *(3, 224, 224))  #.to(device)    #  requires_grad=True

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "resnet34cpu.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=14,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}})
    print('Model has been converted to ONNX')

if __name__ == '__main__':
    Convert_ONNX()
