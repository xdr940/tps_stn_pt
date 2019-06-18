# encoding: utf-8

import os
import torch
import random
import argparse
import mnist_model
import data_loader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from mnist_model import STNClsNet,ClsNet

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 128)
parser.add_argument('--test-batch-size', type = int, default = 128)
parser.add_argument('--epochs', type = int, default = 4)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--momentum', type=float, default = 0.5)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--log-interval', type = int, default = 10)
parser.add_argument('--save-interval', type = int, default = 100)
parser.add_argument('--model', default='no_stn')
parser.add_argument('--angle', type = int, default=60)
parser.add_argument('--span_range', type = int, default = 0.9)
parser.add_argument('--grid_size', type = int, default = 4)
parser.add_argument('--print-freq',type = int,default= 100)
parser.add_argument('--save-freq',type = int,default= 2)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

args.span_range_height = args.span_range_width = args.span_range
args.grid_height = args.grid_width = args.grid_size
args.image_height = args.image_width = 28

# create model
if args.model == 'no_stn':
    print('create model without ClsNet')
    model = ClsNet().to(device)
else:
    print('create model with STN')
    model = STNClsNet(args).to(device)

model = torch.nn.DataParallel(model)


optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
train_loader = data_loader.get_train_loader(args)
test_loader = data_loader.get_test_loader(args)

def train(epoch):
    def print_acc(epoch):
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.data.item()))
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            print_acc(epoch)



def validate():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data,target = data.to(device),target.to(device)

        output = model(data)
        test_loss += F.nll_loss(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.02f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy,
    ))
    log_file.write('{:.02f}\n'.format(accuracy))
    log_file.flush()
    os.fsync(log_file)



def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        #save
        if epoch % args.save_freq == 0:
            checkpoint_path = checkpoint_dir + 'epoch%03d_.pth' % epoch
            torch.save(model.state_dict(), checkpoint_path)

        validate()



if __name__=="__main__":

    checkpoint_dir = 'checkpoint/%s_angle%d_grid%d/' % (
        args.model, args.angle, args.grid_size,
    )
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.isdir('accuracy_log'):
        os.makedirs('accuracy_log')
    log_file_path = 'accuracy_log/%s_angle%d_grid%d.txt' % (
        args.model, args.angle, args.grid_size,
    )

    with open(log_file_path, 'w') as log_file:

        main()