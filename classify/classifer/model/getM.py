from .alex import *
from .efficient import *
from .efficientv2 import *
from .gooleNet import *
from .mobilev2 import *
from .mobilev3 import *
from .regnet import *
from .resnet import *
from .senet import *
from .vgg import *
from .mobileViT.m import *

def getModel(model, cn, usePre, useBefor):
    if model=='alex':
        model = AlexNet(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('alex.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='efficient':
        model = efficientnet_b0(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('efficient.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='efficientv2':
        model = efficientnetv2_s(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('efficientv2.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='gooleNet':
        model = GoogLeNet(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('gooleNet.oth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='mobilev2':
        model = MobileNetV2(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('mobilev2.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='mobilev3':
        model = mobilenet_v3_small(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('mobilev3.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='regnet':
        model = create_regnet(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('regnet.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='resnet':
        model = resnet34(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('resnet.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='senet':
        model = se_resnet_34(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('senet.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='vgg':
        model = vgg('vgg16')
        if usePre: 
            pretrained_dict = torch.load('vgg16.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    elif model=='mobileViT':
        model = mobile_vit_xx_small(num_classes=cn)
        if usePre: 
            pretrained_dict = torch.load('mobileViT.pth')
            model.load_state_dict(pretrained_dict, strict=False)
        elif useBefor:
            pretrained_dict = torch.load(useBefor)
            model.load_state_dict(pretrained_dict, strict=False)
    return model