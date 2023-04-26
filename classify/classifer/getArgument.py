import argparse


def parse_opt():
    # python cTrain.py --model alex --trainData 'data/level0/train' --testData 'data/level0/test' --optimizer 'Adam' --lr 0.0001 --saveName 'al01' --batchSize 20 --epoch 50
    parser = argparse.ArgumentParser(description='命令行中传入一个数字')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--trainData', type=str, default='')
    parser.add_argument('--testData', type=str, default='')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--lr', type=float, default='0.0002')
    parser.add_argument('--saveName', type=str, default='train')
    parser.add_argument('--batchSize', type=int, default='20')
    parser.add_argument('--epoch', type=int, default='50')
    args = parser.parse_args()
    # print(type(args))
    return args


if __name__ == "__main__":
    ops = parse_opt()

