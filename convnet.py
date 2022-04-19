#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init
from skimage import segmentation

np.random.seed(0)

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(
    description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int,
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int,
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int,
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int,
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=10000, type=int,
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=100, type=float,
                    help='compactness of superpixels')
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=5, type=float,
                    help='step size for continuity loss')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int,
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--batch', metavar='B', default=True, type=float,
                    help='whether or not to use batch normalization after final convolution')
parser.add_argument('--version', metavar='V', default="continuity",
                    help='whether to use pixel continuity or superpixel loss')
args = parser.parse_args()

# CNN model


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append(
                nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(args.nChannel))
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(args.nConv-1):
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

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

# continuity loss definition
if args.version == "continuity":
    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)

    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], args.nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, args.nChannel)
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

# superpixel loss definition (slic)
l_inds = []
if args.version == "superpixel":
    labels = segmentation.slic(
        im, compactness=args.compactness, n_segments=args.num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

# train
model = MyNet(data.size(1))
if use_cuda:
    model.cuda()
model.train()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255, size=(100, 3))

# forward pass
for batch_idx in range(args.maxIter):
    optimizer.zero_grad()
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)

    _, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))

    if args.visualize:
        im_target_rgb = np.array(
            [label_colours[c % args.nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        cv2.imshow("output", im_target_rgb)
        cv2.waitKey(10)

    if args.version == "superpixel":
        for i in range(len(l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
        target = torch.from_numpy(im_target)
        if use_cuda:
            target = target.cuda()
        target = Variable(target)
        loss = loss_fn(output, target)

    if args.version == "continuity":
        # continuity loss
        outputHP = output.reshape((im.shape[0], im.shape[1], args.nChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)
        loss = args.stepsize_sim * \
            loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)

    loss.backward()
    optimizer.step()

    print(batch_idx, '/', args.maxIter, '|',
          ' label num :', nLabels, ' | loss :', loss.item())

    if batch_idx == args.maxIter - 1:
        final = im_target.reshape(
            im.shape[0:2])
        if args.batch:
            final.tofile(("single_predictions/batch/" +
                         (args.input).strip('.pngjp') + "_" + args.version + "_" + str(args.nConv) + "_" + str(args.stepsize_sim) + "_" + str(args.stepsize_con) + ".csv"), sep=",")
        else:
            final.tofile(("single_predictions/nobatch/" +
                         (args.input).strip('.pngjp') + "_" + args.version + "_" + str(args.nConv) + "_" + str(args.stepsize_sim) + "_" + str(args.stepsize_con) + ".csv"), sep=",")

    if nLabels <= args.minLabels:
        print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        if args.batch:
            final.tofile(("single_predictions/batch/" +
                         (args.input).strip('.pngjp') + "_" + args.version + "_" + str(args.nConv) + "_" + str(args.stepsize_sim) + "_" + str(args.stepsize_con) + ".csv"), sep=",")
        else:
            final.tofile(("single_predictions/nobatch/" +
                         (args.input).strip('.pngjp') + "_" + args.version + "_" + str(args.nConv) + "_" + str(args.stepsize_sim) + "_" + str(args.stepsize_con) + ".csv"), sep=",")
        break

# save output image
if not args.visualize:
    output = model(data)[0]
    output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel)
    ignore, target = torch.max(output, 1)
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array(
        [label_colours[c % args.nChannel] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)

if args.batch:
    cv2.imwrite(("single_predictions/continuity/" +
                 (args.input).strip('.pngjg') + "_" + str(args.nConv) + "_" + str(args.stepsize_sim) + "_" + str(args.stepsize_con) + ".png"), im_target_rgb)
else:
    cv2.imwrite(("single_predictions/continuity_nobatch/" +
                 (args.input).strip('.pngjg') + "_" + str(args.nConv) + "_" + str(args.stepsize_sim) + "_" + str(args.stepsize_con) + ".png"), im_target_rgb)
