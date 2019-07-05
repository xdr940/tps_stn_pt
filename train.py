# encoding: utf-8

import os
import torch
import random
import argparse
import data_loader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import STNClsNet,ClsNet,NeuralNet
from torchvision import datasets, transforms
from util import AverageMeter
from tensorboardX import SummaryWriter
import datetime

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 128)
parser.add_argument('--test-batch-size', type = int, default = 128)
parser.add_argument('--epochs', type = int, default = 15)
parser.add_argument('--epoch-size', default=15, type=int)
parser.add_argument('--dataset', default='mnist', type=str)


parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--momentum', type=float, default = 0.5)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--log-interval', type = int, default = 10)
parser.add_argument('--save-interval', type = int, default = 100)
parser.add_argument('--model', choices=['unbounded_stn','bounded_stn','no_stn','bp'],default='unbounded_stn')
parser.add_argument('--angle', type = int, default=60)
parser.add_argument('--span_range', type = int, default = 0.9)
parser.add_argument('--grid_size', type = int, default = 4)
parser.add_argument('--print-freq',type = int,default= 100)
parser.add_argument('--save-freq',type = int,default= 2)
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('--nums-worker',type=int,default=8)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

args.span_range_height = args.span_range_width = args.span_range
args.grid_height = args.grid_width = args.grid_size# 4
args.image_height = args.image_width = 28



n_iter = 0
n_iter_val =0
# create model
if args.model == 'no_stn':
    print('create model without ClsNet')
    model = ClsNet().to(device)
elif args.model =='bp':
    print('create model bp')
    model =NeuralNet().to(device)
else:
    print('create model with STN')
    model = STNClsNet(args).to(device)

#model = torch.nn.DataParallel(model)


optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

train_set = datasets.MNIST(
            '/home/roit/datasets/mnist/',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.Lambda(lambda image: image.rotate(random.random() * args.angle * 2 - args.angle)),
                transforms.ToTensor(),
            ]),
        )

test_set = datasets.MNIST(
            '/home/roit/datasets/mnist/',
            train = False,
            download = True,
            transform = transforms.Compose([
                transforms.Lambda(lambda image: image.rotate(random.random() * args.angle * 2 - args.angle)),
                transforms.ToTensor(),
            ]),
        )



train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.nums_worker,
        pin_memory = True if args.cuda else False,
    )

test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.nums_worker,
        pin_memory = True if args.cuda else False,
    )


criterion = nn.CrossEntropyLoss()

def train(args, train_loader, model, train_writer):
    global n_iter
    epoch_losses = AverageMeter()  # 自定义类只能放到cpu内存?
    epoch_acc = AverageMeter()

    for batch_idx, (images, labels) in enumerate(train_loader):
        if args.model =='bp':
            images = images.view(images.size(0), -1)
        images,labels = images.to(device),labels.to(device)
        # forwardpass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()

        # update
        epoch_losses.update(loss.item(), args.batch_size)
        epoch_acc.update(accuracy.item(), args.batch_size)

        # print
        if (batch_idx + 1) % args.print_freq == 0:  # print freq
            print('Step [{}/{}], train_batch_Loss: {:.4f}, train_batch_Acc: {:.2f}'
                  .format(batch_idx + 1, len(train_loader), loss, accuracy))

        # writer
        train_writer.add_scalar('batch_data/batch_loss', loss.item(), n_iter + 1)
        train_writer.add_scalar('batch_data/batch_acc', accuracy.item(), n_iter + 1)

        n_iter += 1

    return epoch_losses.avg, epoch_acc.avg  # list


def validate(args, val_loader, model):
    val_epoch_losses = AverageMeter()
    val_epoch_acc = AverageMeter()
    global n_iter_val


    for batch_idx,(images,labels) in enumerate(val_loader):
        if args.model =='bp':
            images = images.view(images.size(0), -1)
        images, labels = images.to(device), labels.to(device)

       #forwardpass
        outputs = model(images)


        loss = criterion(outputs, labels)

        # NO Backward and optimize!!

        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()


        n_iter_val+=1



        val_epoch_losses.update(loss.item(), args.batch_size)
        val_epoch_acc.update(accuracy.item(), args.batch_size)


    return val_epoch_losses.avg,val_epoch_acc.avg# list


def main():
    # checkpoints and model_args, 标准存储checkpoint
    save_path = '{},{}epochs{},b{},lr{}'.format(
        args.model,
        args.epochs,
        ',epochSize' + str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:  # 保存with时间
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(args.dataset + '_checkpoints', save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # tensorboardX-writer of train,test,output
    train_writer = SummaryWriter(
        os.path.join(save_path, 'train'))  # 'KITTI_occ/05-29-11:36/flownets,adam,300epochs,epochSize1000,b8,lr0.0001'

    test_writer = SummaryWriter(os.path.join(save_path, 'test'))  #



    # dummy_input = torch.rand(13,1,28,28)
    # with SummaryWriter(comment='MyModel') as w:
    #    w.add_graph(model,(dummy_input,))

    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, train_loader, model, train_writer)
        with torch.no_grad():
            val_loss, val_acc = validate(args, train_loader, model)
        # epoch-record data log
        train_writer.add_scalar(tag='epoch_data/epoch_loss', scalar_value=train_loss, global_step=epoch)
        train_writer.add_scalar(tag='epoch_data/epoch_acc', scalar_value=train_acc, global_step=epoch)

        test_writer.add_scalar(tag='epoch_data/epoch_loss', scalar_value=val_loss, global_step=epoch)
        test_writer.add_scalar(tag='epoch_data/epoch_acc', scalar_value=val_acc, global_step=epoch)
        '''
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            training_writer.add_histogram(tag, value.data.cpu().numpy(), n_iter + 1)
            training_writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(),n_iter + 1)
        '''
        # 3. Log training images (image summary)

        # print epoch
        print('epoch [{}/{}]\n'
              'train: avg_loss: {:.4f}, avg_acc: {:.2f}'
              '\nvalidate: avg_loss: {:.4f},avg_acc:{:.2f}'
              .format(epoch + 1, args.epochs,
                      train_loss, train_acc,
                      val_loss, val_acc))

        # model save
        torch.save(model, 'model.pth.rar')

        # csv epoch data record
        '''
        with open(args.save_path / args.log_summary, 'a') as csvfile:  # 每个epoch留下结果
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])  # 第二个就是validataion 中的epoch-record
            # loss<class 'list'>: ['Total loss', 'Photo loss', 'Exp loss']
        '''
    train_writer.close()
    test_writer.close()

    return 0


if __name__ == "__main__":
    main()