'''Train CIFAR100 with PyTorch.'''
import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

from shb import SHB
from nshb import NSHB
from models.resnet import resnet18
from models.wideresnet import WideResNet28_10
from models.mobilenetv2 import mobilenetv2
from utils import progress_bar

norm_list = []
steps = 0
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    p_norm = 0
    global norm_list
    global steps
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if args.method == "sampling" and steps == args.steps:
            mt_list = optimizer.step(itr=steps)
            full_list = get_full_grad_list(net,trainset,optimizer)
            for m, g in zip(mt_list, full_list):
                p_norm += torch.sum(torch.mul(m-g, m-g))
            p_norm = torch.sqrt(p_norm)
            wandb.log({'noise_norm': p_norm})
            sys.exit()
        else:
            optimizer.step()
            steps += 1
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if args.method == "critical":
            norm = 0
            parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                norm += param_norm.item() ** 2
            norm = norm ** 0.5
            norm_ave = norm_work(norm_list, norm)

            wandb.log({'g-norm': norm,
                       'ave-norm': norm_ave})
            
            if norm_ave <= args.eps and epoch > 10:
                print(steps, norm_ave)
                wandb.log({'steps': steps})
                sys.exit()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    training_acc = 100.*correct/total
    wandb.log({'training_acc': training_acc,
               'training_loss': train_loss/(batch_idx+1)})

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    wandb.log({'accuracy': acc}) 

def get_full_grad_list(net, train_set, optimizer):
    parameters=[p for p in net.parameters()]
    batch_size=1000
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    device='cuda:0'
    init=True
    full_grad_list=[]

    for i, (xx,yy) in (enumerate(train_loader)):
        xx = xx.to(device, non_blocking = True)
        yy = yy.to(device, non_blocking = True)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
        loss.backward()
        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init=False

        for i, params in enumerate(parameters):
            g = params.grad.detach().data
            full_grad_list[i] += (batch_size / len(train_set)) * g
    return full_grad_list

def norm_work(norm_list, norm):
    norm_list.append(norm)
    average = sum(norm_list) / len(norm_list)
    return average

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default="CIFAR100", type=str, help="CIFAR100, CIFAR10")
    parser.add_argument('--model', default="ResNet18", type=str, help="ResNet18, WideResNet-28-10, MobileNetv2")
    parser.add_argument('--optimizer', default="shb", type=str, help="sgd, shb, nshb")
    parser.add_argument('--method', default="normal", type=str, help="normal, critical, sampling")
    parser.add_argument('--epochs', default=200, type=int, help="the number of epochs")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=8, type=int, help='training batch size')
    parser.add_argument('--beta1', default=0.9, type=float, help="effective momentum value")
    parser.add_argument('--nu', default=0.5, type=float, help='effective nu value')
    parser.add_argument('--eps', default=0.5, type=float, help='the stopping condition for a method "critical"')
    parser.add_argument('--steps', default=10000, type=int, help='the number of steps for a method "sampling"')
    
    args = parser.parse_args()
    wandb_project_name = "role_of_momentum"
    wandb_exp_name = f"{args.method},{args.optimizer},b={args.batchsize},lr={args.lr},{args.repeat}"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "XXXXXX")
    wandb.init(settings=wandb.Settings(start_method='fork'))

    print('==> Preparing data..')
    if args.dataset == "CIFAR100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(15),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif args.dataset == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    device = 'cuda:0'
    if args.model == "ResNet18":
        net = resnet18()
    elif args.model == "WideResNet-28-10":
        net = WideResNet28_10()
    elif args.model == "MobileNetv2":
        net = mobilenetv2()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0)
    elif args.optimizer == "shb":
        optimizer = SHB(net.parameters(), lr=args.lr, momentum=args.beta1)
    elif args.optimizer == "nshb":
        optimizer = NSHB(net.parameters(), lr=args.lr, momentum=args.beta1)
    print(optimizer)

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)