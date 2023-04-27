import torch
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

jit_model = torch.jit.load('yolov8m.torchscript')
# jit_model = jit_model

img = Image.open('1.jpg') 

transform = transforms.Compose([
    transforms.CenterCrop((640, 640)),
    transforms.ToTensor()
])

t = transform(img).to(device)

# print(t.device.type)

while True:
    print('cccc')
    results = jit_model(t.unsqueeze(0))  # predict on an image

