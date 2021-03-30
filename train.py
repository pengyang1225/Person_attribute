"""
author:guopei
"""

import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import transforms
from tensorboardX import SummaryWriter
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from criterion import LSR
from models import resnest50

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default="resnest50", help='net type')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-e', type=int, default=50, help='training epoches')
    parser.add_argument('-warm', type=int, default=1, help='warm up phase')
    parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
    args = parser.parse_args()

    #checkpoint directory
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    #tensorboard log directory
    log_path = os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    #get dataloader
    train_transforms = transforms.Compose([
        transforms.ToCVImage(),
        #transforms.RandomResizedCrop(settings.IMAGE_SIZE),
        transforms.Resize(settings.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        #transforms.RandomErasing(),
        #transforms.CutOut(56),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.ToCVImage(),
        #transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.Resize(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])

    train_dataloader = get_train_dataloader(
        settings.DATA_PATH,
        train_transforms,
        args.b,
        args.w
    )

    test_dataloader = get_test_dataloader(
        settings.DATA_PATH,
        test_transforms,
        int(args.b/8),
        args.w
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #net = get_network(args)
    net = resnest50(pretrained=False)
    net = init_weights(net)
    # load pretraines model
    weights = torch.load("pretrain_models/resnest50-528c19ca.pth")
    net_dict = net.state_dict()
    for k,v in weights.items():
        if k in weights.items():
            net_dict[k] = v
    net.load_state_dict(net_dict)
    print("=>load model form pretrain_models/resnest50-528c19ca.pth")

    
    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    
    net = nn.DataParallel(net, device_ids=args.gpus)
    net = net.cuda()

    #visualize the network
    #visualize_network(writer, net.module)

    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
    #lsr_loss = LSR()

    #apply no weight decay on bias
    params = split_weights(net)
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    #set up warmup phase learning rate scheduler
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #set up training phase learning rate scheduler
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES)
    #train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.e - args.warm)

    best_acc = 0.0
    for epoch in range(1, args.e + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        #training procedure
        net.train()

        for batch_index, (images, labels) in enumerate(train_dataloader):
            if epoch <= args.warm:
                warmup_scheduler.step()

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            predicts = net(images)
            #loss = lsr_loss(predicts, labels)
            # print(" predicts[0]", predicts[0])
            # print(" labels[:, 0]", labels[:, 0])
            # print(" predicts[1]", predicts[1])
            # print(" labels[:, 1]", labels[:, 1])

            loss_gender = cross_entropy(predicts[0], labels[:, 0].long())
            loss_age = cross_entropy(predicts[1], labels[:, 1].long())
            loss_orientation = cross_entropy(predicts[2], labels[:, 2].long())
            loss_hat = cross_entropy(predicts[3], labels[:, 3].long())
            loss_glasses = cross_entropy(predicts[4], labels[:, 4].long())
            loss_handBag = cross_entropy(predicts[5], labels[:, 5].long())
            loss_shoulderBag = cross_entropy(predicts[6], labels[:, 6].long())
            loss_backBag = cross_entropy(predicts[7], labels[:, 7].long())
            loss_upClothing = cross_entropy(predicts[8], labels[:, 8].long())
            loss_downClothing= cross_entropy(predicts[9], labels[:, 9].long())

            loss = loss_gender + loss_age + loss_orientation + loss_hat + loss_glasses + loss_handBag + loss_shoulderBag + loss_backBag + loss_upClothing + loss_downClothing
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

            if batch_index % 10 == 0:
                print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLoss_gender: {:0.4f}\tLoss_age: {:0.4f}\tLoss_ori: {:0.4f}\tLoss_hat: {:0.4f}\tLoss_glasses: {:0.4f}\tLoss_handBag: {:0.4f}\t'.format(
                    loss.item(),
                    loss_gender.item(),
                    loss_age.item(),
                    loss_orientation.item(),
                    loss_hat.item(),
                    loss_glasses.item(),
                    loss_handBag.item(),
                    epoch=epoch,
                    trained_samples=batch_index * args.b + len(images),
                    total_samples=len(train_dataloader.dataset),
                ))

            #visualization
            visualize_lastlayer(writer, net, n_iter)
            visualize_train_loss(writer, loss.item(), n_iter)


        visualize_learning_rate(writer, optimizer.param_groups[0]['lr'], epoch)
        visualize_param_hist(writer, net, epoch) 

        net.eval()

        total_loss = 0
        correct = np.zeros(10)
        ignore = np.zeros(10)
        print("=>test model")
        for images, labels in tqdm(test_dataloader):

            images = images.cuda()
            labels = labels.cuda()

            predicts = net(images)

            for index in range(10):

                _, preds = predicts[index].max(1)
                ignore[index] += int((labels[:, index]==-1).sum())
                correct[index] += preds.eq(labels[:, index]).sum().float()

                loss = cross_entropy(predicts[index], labels[:, index].long())
                total_loss += loss.item()

        test_loss = total_loss / len(test_dataloader)
        all_list = np.array([len(test_dataloader.dataset) for i in range(10)])-ignore
        acc_list = correct / all_list
        print(acc_list.tolist())
        print("gender_acc:%.4f, age_acc:%.4f, orientation_acc:%.4f, hat_acc:%.4f, glasses_acc:%.4f, handBag_acc:%.4f, shoulderBag_acc:%.4f, backBag_acc:%.4f, upClothing_acc:%.4f, downClothing_acc:%.4f" % tuple(acc_list))

        acc = float(acc_list.mean())

        print('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
        print()

        visualize_test_loss(writer, test_loss, epoch)
        visualize_test_acc(writer, acc, epoch)

        # save weights file
        if best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
 #python3 ./.conda/envs/pytorch-tensorrt/lib/python3.7/site-packages/tensorboard/main.py  --logdir=/home/py/code/mana/attribute/runs
