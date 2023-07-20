import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

import attack
import data_loaders
from functions import *
from models import *
from utils import train, val

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=2, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd',type=str,help='model')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg11',type=str,help='model')
parser.add_argument('-T','--time',default=8, type=int,metavar='N',help='snn simulation time, set 0 as ANN')
parser.add_argument('-tau','--tau',default=1., type=float,metavar='N',help='leaky constant')
parser.add_argument('-en', '--encode', default='constant', type=str, help='(constant/poisson)')

# training configuration
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-wd','--wd',default=5e-4, type=float,help='weight decay')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    dvs = False
    if args.dataset.lower() == 'cifar10':
        use_cifar10 = True
        num_labels = 10
    elif args.dataset.lower() == 'cifar100':
        use_cifar10 = False
        num_labels = 100
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
    elif args.dataset.lower() == 'dvscifar':
        num_labels = 10
        assert args.time == 10
        dvs = True
    elif args.dataset.lower() == 'dvsgesture':
        num_labels = 11
        assert args.time == 10
        dvs = True
        init_s = 64
    elif args.dataset.lower() == 'nmnist':
        num_labels = 10
        assert args.time == 10
        dvs = True
        init_s = 34

    #>>>>>>>IMPORTANT<<<<<<<< Edit log_dir
    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    if 'dvsgesture' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvsgesture(root='/home/butong/datasets/DVSGesture/')
    elif 'dvscifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_dvscifar(root='/home/butong/datasets/CIFAR10DVS/')
    elif 'nmnist' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_nmnist(root='/home/butong/datasets/NMNIST/')
    elif 'cifar' in args.dataset.lower():
        train_dataset, val_dataset, znorm = data_loaders.build_cifar(use_cifar10=use_cifar10)
    elif args.dataset.lower() == 'svhn':
        train_dataset, val_dataset, znorm = data_loaders.build_svhn()
    else:
        raise AssertionError("data not supported")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    if 'cnndvs' in args.model.lower():
        model = CNNDVS(args.time, num_labels, args.tau, 2, init_s)
    elif 'vggdvs' in args.model.lower():
        model = VGGDVS(args.model.lower(), args.time, num_labels, znorm, args.tau)
    elif 'vgg' in args.model.lower():
        model = VGG(args.model.lower(), args.time, num_labels, znorm, args.tau)
    elif 'resnet17' in args.model.lower():
        model = ResNet17(args.time, args.tau, num_labels, znorm)
    elif 'resnet19' in args.model.lower():
        model = ResNet19(args.time, args.tau, num_labels, znorm)
    else:
        raise AssertionError("model not supported")

    model.set_simulation_time(args.time)
    model.to(device)
    model.poisson = (args.encode.lower() == 'poisson')

    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0

    # IMPORTANT<<<<<<<<<<<<< modifed
    identifier = args.model
    identifier += '_T[%d]'%(args.time)
    identifier += '_tau[%.2f]'%(args.tau)
    if args.encode == 'poisson':
        identifier += "_poisson"
    identifier += args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    logger.info('start training!')
    
    for epoch in range(args.epochs):
        loss, acc = train(model, device, train_loader, criterion, optimizer, args.time, dvs=dvs)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc))
        scheduler.step()
        tmp = val(model, test_loader, device, args.time, dvs)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch , args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()