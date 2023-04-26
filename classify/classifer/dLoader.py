import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# 自定义数据集的类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_list[idx])
        img = Image.open(img_path).convert('RGB')
        label = int(self.img_list[idx].split('.')[0])
        if self.transform:
            img = self.transform(img)
        return img, label

# 定义转换（归一化）和加载数据集
transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trainset = CustomDataset(root_dir='./train', transform=transform)
trainloader = DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = CustomDataset(root_dir='./test', transform=transform)
testloader = DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# 定义类别标签
classes = ('label1', 'label2', 'label3', 'label4', 'label5')
