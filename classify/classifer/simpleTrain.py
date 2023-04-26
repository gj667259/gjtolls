import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from tqdm import tqdm
import os
import sys


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义数据预处理
transform = transforms.Compose(
    [  transforms.Resize((450,550)),
    # transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
nw = min([os.cpu_count(), 8])


# 加载训练集
trainset = torchvision.datasets.ImageFolder(root='./data/level0/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=nw)


# 加载测试集
testset = torchvision.datasets.ImageFolder(root='./data/level0/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=nw)


# 定义类别标签
classes = ('level0_ng', 'level0_ok')


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


net = AlexNet(num_classes=2).to(device)

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

epochs = 10
save_path = './AlexNet.pth'
best_acc = 0.0
train_steps = len(trainloader)
# 训练网络
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(trainloader, file=sys.stdout)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

print('Finished Training')

# 测试网络
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10 test images: %d %%' % (100 * correct / total))

# 输出每个类别的准
