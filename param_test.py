#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init

# continuity only, superpixel performed poorly

np.random.seed(0)

use_cuda = torch.cuda.is_available()

nChannel = 100
maxIter = 1000
minLabels = 3
lr = 0.1
nConv = [4, 5, 6, 7]
stepsize_sim = [1.5, 2, 2.5, 3]
stepsize_con = [1, 1, 5, 2]

parser = argparse.ArgumentParser(
    description='PyTorch Unsupervised Segmentation')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--batch', metavar='B', default=True, type=float,
                    help='whether or not to use batch normalization after final convolution')
args = parser.parse_args()


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim, num_Conv):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(num_Conv-1):
            self.conv2.append(
                nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))
        self.conv3 = nn.Conv2d(nChannel, nChannel,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x, nConv):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        if args.batch:
            x = self.bn3(x)
        return x


# load image
im = cv2.imread(args.input)
data = torch.from_numpy(
    np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
if use_cuda:
    data = data.cuda()
data = Variable(data)

for nc in range(len(nConv)):
    for ss in range(len(stepsize_sim)):
        for sc in range(len(stepsize_con)):
            if stepsize_sim[ss] == stepsize_con[sc]:
                continue
            # train
            model = MyNet(data.size(1), nConv[nc])
            if use_cuda:
                model.cuda()
            model.train()

            # similarity loss definition
            loss_fn = torch.nn.CrossEntropyLoss()

            # continuity loss definition
            loss_hpy = torch.nn.L1Loss(size_average=True)
            loss_hpz = torch.nn.L1Loss(size_average=True)

            HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], nChannel)
            HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, nChannel)
            if use_cuda:
                HPy_target = HPy_target.cuda()
                HPz_target = HPz_target.cuda()

            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            label_colours = np.random.randint(255, size=(100, 3))

            for batch_idx in range(maxIter):
                # forwarding
                optimizer.zero_grad()
                output = model(data, nConv[nc])[0]
                output = output.permute(
                    1, 2, 0).contiguous().view(-1, nChannel)

                outputHP = output.reshape((im.shape[0], im.shape[1], nChannel))
                HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
                HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
                lhpy = loss_hpy(HPy, HPy_target)
                lhpz = loss_hpz(HPz, HPz_target)

                ignore, target = torch.max(output, 1)
                im_target = target.data.cpu().numpy()
                nLabels = len(np.unique(im_target))
                im_target_rgb = np.array(
                    [label_colours[c % nChannel] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(
                    im.shape).astype(np.uint8)

                # loss
                loss = stepsize_sim[ss] * \
                    loss_fn(output, target) + stepsize_con[sc] * (lhpy + lhpz)

                loss.backward()
                optimizer.step()

                print(batch_idx, '/', maxIter, '|',
                      ' label num :', nLabels, ' | loss :', loss.item())

                if nLabels <= minLabels:
                    print("nLabels", nLabels, "reached minLabels", minLabels, ".")
                    break

            if args.batch:
                cv2.imwrite(("single_predictions/continuity/" +
                             (args.input).strip('.pngjg') + "_" + str(nConv[nc]) + "_" + str(stepsize_sim[ss]) + "_" + str(stepsize_con[sc]) + ".png"), im_target_rgb)
            else:
                cv2.imwrite(("single_predictions/continuity_nobatch/" +
                             (args.input).strip('.pngjg') + "_" + str(nConv[nc]) + "_" + str(stepsize_sim[ss]) + "_" + str(stepsize_con[sc]) + ".png"), im_target_rgb)
