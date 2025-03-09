import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.MGTANet_models import MGTANet
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou
import pytorch_fm
import pytorch_ssim

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./MGTANet-logs/ORSSD')
from data import test_dataset

torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=45, help='every n epochs decay learning rate')
parser.add_argument('--testsize', type=int, default=384, help='testing size')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = MGTANet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.NAdam(params, opt.lr)
#
image_root = './dataset/ORSSD/train/image/'
gt_root = './dataset/ORSSD/train/gt/'
depth_root = './dataset/ORSSD/train/depth/'

train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)


def only_iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = IOU(pred, target)

    loss = iou_out + ssim_out

    return loss


def bce_iou_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    bce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return (bce + iou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    total_loss = 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, depths = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        images = images.cuda()
        gts = gts.cuda()
        depths = depths.cuda()

        pred1, pred2, pred3, pred4 = model(images, depths)
        # bce+iou+fmloss
        loss1 = CE(pred1, gts) + only_iou_loss(pred1, gts)
        loss2 = CE(pred2, gts) + only_iou_loss(pred2, gts)
        loss3 = CE(pred3, gts) + only_iou_loss(pred3, gts)
        loss4 = CE(pred4, gts) + only_iou_loss(pred4, gts)
        # loss3 = CE(s3, gts) + IOU(s3_sig, gts) + floss(s3_sig, gts)

        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                       loss2.data))
        total_loss += loss.item()
    # ORSSD
    writer.add_scalar('Loss/train', total_loss / 600, epoch)
    print('第{}个epoch的loss: {:.4f}'.format(epoch, total_loss / 600))

    save_path = 'models/MGTANet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + 'MGTANet.pth' + '.%d' % epoch)



print("Let's go!")

if __name__ == '__main__':
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
    writer.close()
