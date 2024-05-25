import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb

from shb import SHB
from models.resnet import resnet18
from models.wideresnet import WideResNet28_10
from utils import progress_bar

#norm_list = []
#steps = 0
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    #global norm_list
    #global steps
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        norm = 0
        parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            norm += param_norm.item() ** 2
        norm = norm ** 0.5
        
        wandb.log({'norm': norm})

        #if norm_ave <= 0.5:
        #    norm_ave = norm_work(norm_list, norm)
        #    steps += 1
        #    print(steps, norm_ave)
        #    wandb.log({'steps': steps,
        #               'ave-norm': norm_ave})
        #    sys.exit()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        if args.method in ["lr", "hybrid", "cosine", "poly", "exp"]:
            last_lr = scheduler.get_last_lr()[0]
            wandb.log({'last_lr': last_lr})

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

def norm_work(norm_list, norm):
    norm_list.append(norm)
    average = sum(norm_list) / len(norm_list)
    return average

def next_gamma(power, M, m):
    #power means the power of polynomial decay
    #M means the number of epochs
    #m means the current epoch
    top = (M - m)
    bottom = (M - (m-1))
    gamma = (top/bottom) ** power
    return gamma

def lr_poly(lr, M, m, power):
    gamma = (M - m)/(M - (m - 1))
    gamma = gamma ** (power / 2)
    next_lr = lr * gamma
    return next_lr

def batch_poly(batch, M, m, power, balance):
    gamma = (M - (m - 1))/(M - m)
    gamma = gamma ** (power * balance)
    next_batch = batch * gamma
    return next_batch

def next_delta_SHB(lr, batch, beta, C, K):
    #lr is current learning rate
    #batch is current batch size
    #beta is current momentum factor
    #C means the variance of stochastic gradient
    #K means the upper bound of gradient norm
    beta_hat = (beta * (beta ** 2 - beta + 1)) / ((1 - beta) ** 2)
    inside = C / batch + beta_hat * ((C / batch) + K ** 2)
    delta = lr * np.sqrt(inside)
    return delta

def next_delta_NSHB(lr, batch, beta, C, K):
    inside = C / batch + (4 * (beta ** 2) * ((C / batch) + K ** 2))
    delta = lr * np.sqrt(inside)
    return delta

def solve_3eq(uni, beta):
    #uni is in coefficient for cubic equation
    #beta is current momentum factor
    roots = np.roots([1, (-1 * (1 + uni)), (1 + (2 * uni)), (-1 * uni)])
    real_roots = roots[np.isreal(roots)].real
    filtered_roots = [root for root in real_roots if root <= beta]
    next_beta = max(max(filtered_roots), 0)
    return next_beta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batchsize', default=256, type=int, help='training batch size')
    parser.add_argument('--epochs', default=200, type=int, help="the number of epochs")
    parser.add_argument('--power', default=0.9, type=float, help="polinomial or exponential power")
    parser.add_argument('--method', default="constant", type=str, help="constant, lr, beta, lr-batch, beta-batch, lr-beta, cosine, exp")
    parser.add_argument('--optimizer', default="shb", type=str, help="sgd, shb, nshb")
    parser.add_argument('--model', default="ResNet18", type=str, help="ResNet18, WideResNet-28-10")
    parser.add_argument('--beta', default=0.9, type=float, help="effective momentum value")
    
    args = parser.parse_args()
    wandb_project_name = "new-sigma-CIFAR100"
    wandb_exp_name = f"{args.method},{args.optimizer},b={args.batchsize},lr={args.lr}"
    wandb.init(config = args,
               project = wandb_project_name,
               name = wandb_exp_name,
               entity = "XXXXXX")
    wandb.init(settings=wandb.Settings(start_method='fork'))

    print('==> Preparing data..')
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

    device = 'cuda:0'
    if args.model == "ResNet18":
        net = resnet18()
    if args.model == "WideResNet-28-10":
        net = WideResNet28_10()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0)
    elif args.optimizer == "shb":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.beta)
        if args.model == "ResNet18":
            C = 25.318
            K = 1.77
        elif args.model == "WideResNet-28-10":
            C = 0.79
            K = 1.66
    elif args.optimizer == "nshb":
        optimizer = SHB(net.parameters(), lr=args.lr, momentum=args.beta)
        if args.model == "ResNet18":
            C = 128
            K = 4.5
        elif args.model == "WideResNet-28-10":
            C = 1.0
            K = 4.262

    if args.method == "lr":
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=args.power)
    elif args.method == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.method == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.power)
    print(optimizer)

    start_epoch = 0
    next_lr = args.lr
    next_batch = args.batchsize
    next_beta = args.beta

    if args.batchsize == 16: balance = 1.503
    elif args.batchsize == 32: balance = 1.335
    elif args.batchsize == 64: balance = 1.169
    elif args.batchsize == 128: balance = 1.0
    elif args.batchsize == 256: balance = 0.835421888053467
    elif args.batchsize == 512: balance = 0.666666666666667
    elif args.batchsize == 1024: balance = 0.5
    
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
        if args.method in ["lr", "cosine", "exp"]:
            scheduler.step()
        elif args.method in ["beta", "beta-batch", "lr-beta"]:
            wandb.log({'beta1': next_beta})
            old_batch = next_batch
            old_lr = next_lr
            if args.method in ["beta-batch"]:
                wandb.log({'batch': int(next_batch)})
                next_batch = batch_poly(next_batch, args.epochs, epoch, args.power, balance)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(next_batch), shuffle=True, num_workers=2)
            elif args.method in ["lr-beta"]:
                wandb.log({'last_lr': next_lr})
                next_lr = lr_poly(next_lr, args.epochs, epoch, args.power)
            if args.optimizer == "shb":
                top = ((next_gamma(args.power, args.epochs, epoch) * next_delta_SHB(old_lr, old_batch, next_beta, C, K)) ** 2) - ((next_lr ** 2) * (C / next_batch))
                bottom = (next_lr ** 2) * ((K ** 2) + (C / next_batch))
                uni = top / bottom
                next_beta = solve_3eq(uni, next_beta)
                optimizer = optim.SGD(net.parameters(), lr=next_lr, momentum=next_beta)
            elif args.optimizer == "nshb":
                top = ((next_gamma(args.power, args.epochs, epoch) * next_delta_NSHB(old_lr, old_batch, next_beta, C, K)) ** 2) - ((next_lr ** 2) * (C / next_batch))
                bottom = (next_lr ** 2) * 4 * ((K ** 2) + (C / next_batch))
                square = max(top / bottom, 0)
                next_beta = np.sqrt(square)
                optimizer = SHB(net.parameters(), lr=next_lr, momentum=next_beta)
        elif args.method in ["lr-batch"]:
            old_batch = next_batch
            old_lr = next_lr
            wandb.log({'last_lr': next_lr,
                       'batch': int(next_batch)})
            next_batch = batch_poly(next_batch, args.epochs, epoch, args.power, balance)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(next_batch), shuffle=True, num_workers=2)
            if args.optimizer == "shb":
                top = next_gamma(args.power, args.epochs, epoch) * next_delta_SHB(old_lr, old_batch, next_beta, C, K)
                bottom = next_delta_SHB(1, next_batch, next_beta, C, K)
                next_lr = top / bottom
                optimizer = optim.SGD(net.parameters(), lr=next_lr, momentum=next_beta)
            elif args.optimizer == "nshb":
                top = next_gamma(args.power, args.epochs, epoch) * next_delta_NSHB(old_lr, old_batch, next_beta, C, K)
                bottom = next_delta_NSHB(1, next_batch, next_beta, C, K)
                next_lr = top / bottom
                optimiser = SHB(net.parameters(), lr=next_lr, momentum=next_beta)