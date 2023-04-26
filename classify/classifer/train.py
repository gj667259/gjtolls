import os
import sys
import json
import random

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
# import matplotlib.pyplot as plt
# import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import alex

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # net = regnet.create_regnet(num_classes=2).to(device)
    net = alex.AlexNet(num_classes=2).to(device)
    save_path = 'alex2.pth'
    print(save_path)
    batch_size = 100
    dataPath = './data/newfiredata'
    data_transform = {
        # 450 550
        "train": transforms.Compose([transforms.Resize((224,224)),
                        transforms.RandomHorizontalFlip(),
                        #  transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((224,224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    
    nw = min([os.cpu_count(), 8])
    # 加载训练集
    trainset = datasets.ImageFolder(root= dataPath + '/train', transform=data_transform['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=nw)
    # 加载测试集
    testset = datasets.ImageFolder(root= dataPath + '/val', transform=data_transform['val'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=nw)

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    train_num = len(trainset)
    val_num = len(testset)

    label_list = trainset.class_to_idx
    cla_dict = dict((val, key) for key, val in label_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(dataPath + '/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 100
    
    best_acc = 0.0
    train_steps = len(trainloader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(trainloader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()  # 取元素

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(testloader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), dataPath + '/' + save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
