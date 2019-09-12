from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from models import *


import sys
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')

parser.add_argument('--name', type=str, nargs='+', default=['1', 'itervar', '1', 'v100'],
                        help='Dataset name: batchsize, feature type, layer, machine')
parser.add_argument('--ratio', type=float, default=0.1, metavar='RA',
                    help='Training/Test Split')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-decay', type=float, default=0.2,
                    help='learning rate ratio')
parser.add_argument('--lr-schedule', type=str, default='normal',
                    help='learning rate schedule')
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[30,60],
                        help='Decrease learning rate at these epochs.')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--arch', type=str, default='Net',
            help='choose the archtecure')


parser.add_argument('--saving-folder', type=str, default='checkpoint',
            help='choose saving name')
parser.add_argument('--resume', type=str, default='',
            help='choose model')

args = parser.parse_args()
# set random seed to reproduce the work
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))

# get dataset
train_loader, test_loader = getData(name=args.name, ratio=args.ratio, train_bs=args.batch_size, test_bs=args.test_batch_size)

# get model and optimizer
if args.name[1] == 'itervar':
    if args.name[2] == '0':
        _d = 2558 # 1604
    else:
        _d = 1604
else:
    _d=482
model_list = {
    'Net': Net(dim=_d),
}

model = model_list[args.arch].cuda()
model = torch.nn.DataParallel(model)
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

# Remember to implement the rank loss!!!!!!!!!!!!
criterion = nn.MSELoss()
# criterion = rank_loss
# Remember to implement the rank loss!!!!!!!!!!!!

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


##############
# Testing Transfer Learning
##############
if args.resume:
    model.load_state_dict(torch.load(args.resume))
    
if args.resume:
    for name, param in model.named_parameters():
        if 'feature' in name and ('0' in name or '4' in name or '8' in name or '12' in name):
            param.requires_grad = False
    print('remember to forzen begining layers')
    
    
best_test = 1e10
    
for epoch in range(1, args.epochs + 1):
    
    print('Current Epoch: %d/%d' %(epoch, args.epochs))
    
    train(epoch, model, train_loader, optimizer, criterion)    
    test_loss = test(model, test_loader, criterion)   
    
    optimizer=exp_lr_scheduler(epoch, optimizer, strategy=args.lr_schedule, decay_eff=args.lr_decay, decayEpoch=args.lr_decay_epoch)
    
    if test_loss < best_test:
        best_test = test_loss
        saving_name = '/features_'+args.name[1]+'_layer_'+args.name[2]+'_batchsize_'+args.name[0]+'_ratio_'+str(args.ratio)
        torch.save(model.state_dict(), args.saving_folder + saving_name +  '_net.pkl')  

print('Best Testing Loss is: ', best_test)

f=open('result' + '.txt','a+')
if args.ratio == 0.1:
    f.write('/features_'+args.name[1]+'_layer_'+args.name[2]+'_batchsize_'+args.name[0]+'\n')
f.write(str(np.around(best_test, 3)) + ', ')
if args.ratio == 0.9:
    f.write('\n\n')
f.close()
