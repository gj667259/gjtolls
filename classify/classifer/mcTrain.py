import torch
from torchvision import transforms, datasets
import torch.nn as nn
import json
from tqdm import tqdm

import random
import argparse
import os
import sys

from model.getM import getModel 


def parse_opt():
    # python cTrain.py --model alex --trainData 'data/level0/train' --split 1 --nc 2  --optimizer 'Adam' --lr 0.0001 --saveName 'al01' --batchSize 20 --epoch 50
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--trainData', type=str, default='')
    parser.add_argument('--split', type=int, default='')
    parser.add_argument('--testData', type=str, default='')
    parser.add_argument('--nc', type=int, default=1000)
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--lr', type=float, default='0.0002')
    parser.add_argument('--saveName', type=str, default='train')
    parser.add_argument('--batchSize', type=int, default='20')
    parser.add_argument('--epoch', type=int, default='50')
    parser.add_argument('--usePre', type=bool, default=False)
    parser.add_argument('--useBefor', type=str, default='')
    args = parser.parse_args()
    # print(type(args))
    return args

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = args.batchSize
    trainData = args.trainData
    testData = args.testData
    data_transform = {"train": transforms.Compose([transforms.Resize((224,224)),
                        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    
    nw = min([os.cpu_count(), 8])
    
    trainset = datasets.ImageFolder(root= trainData, transform=data_transform['train'])

    if args.testData == '':
        imgLen = len(trainset.imgs)
        spl = args.split
        t = int(imgLen * spl /10)
        random.shuffle(trainset.imgs)
        testset = trainset
        testset.imgs = trainset.imgs[0:t]
        trainset.imgs = trainset.imgs[t: imgLen]
    else:
        testset = datasets.ImageFolder(root= testData, transform=data_transform['val'])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=nw)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=nw)
    train_num = len(trainset)
    val_num = len(testset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    label_list = trainset.class_to_idx
    # {0: 'level0_ng', 1: 'level0_ok'} 与上面反转
    cla_dict = dict((val, key) for key, val in label_list.items())
    
    json_str = json.dumps(cla_dict, indent=4)
    with open(trainData + '/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    loss_function = nn.CrossEntropyLoss()

    assert args.nc != train_num, f'error, classes is not equal'
    
    net = getModel(args.model, args.nc).to(device)

    lr = args.lr
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))  # adjust beta1 to momentum
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)

    best_acc = 0.0
    train_steps = len(trainloader)
    for epoch in range(args.epoch):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(trainloader, file=sys.stdout)
        for s, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, args.epoch, loss)
        net.eval()
        acc = 0.0 
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
            torch.save(net.state_dict(), './' + args.saveName + '.pth')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
