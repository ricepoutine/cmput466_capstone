#from __future__ import print_function
import os
from os.path import dirname, join as pjoin
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
from skimage import segmentation

np.random.seed(0)

use_cuda = torch.cuda.is_available()

# using best hyperparameters from test results on CT image
nChannel = 100
maxIter = 1000
minLabels = 3
lr = 0.1
nConv = 6

files = []
batch = True

version = "continuity"
test_or_ct = "CT"  # CT or test

# continuity settings
stepsize_sim = 1.5
stepsize_con = 1

# superpixel settings
compactness = 100
num_superpixels = 10000

# helper for organizing file names based on batch


def isbatch(batch):
    return "_nobatch" if not batch else ""


if test_or_ct == "CT":
    # load CT images
    data_dir = pjoin(dirname(os.path.realpath(__file__)),
                     'CT')
elif test_or_ct == "test":
    # load test images
    data_dir = pjoin(dirname(os.path.realpath(__file__)),
                     'data', 'images', 'test')


for f in os.listdir(data_dir):
    files.append(f)


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append(
                nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannel))
        self.conv3 = nn.Conv2d(nChannel, nChannel,
                               kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        if batch:
            x = self.bn3(x)
        return x

# test suite:


for file in files:
    # load image
    im = cv2.imread(pjoin(data_dir, file))
    data = torch.from_numpy(
        np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    if version == "continuity":
        loss_hpy = torch.nn.L1Loss(size_average=True)
        loss_hpz = torch.nn.L1Loss(size_average=True)
        HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], nChannel)
        HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, nChannel)
        if use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()

        # superpixel loss definition (slic)
    l_inds = []
    if version == "superpixel":
        labels = segmentation.slic(
            im, compactness=compactness, n_segments=num_superpixels)
        labels = labels.reshape(im.shape[0]*im.shape[1])
        u_labels = np.unique(labels)
        for i in range(len(u_labels)):
            l_inds.append(np.where(labels == u_labels[i])[0])

    # train
    model = MyNet(data.size(1))
    if use_cuda:
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))

    # forward pass
    for batch_idx in range(maxIter):
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        if version == "superpixel":
            for i in range(len(l_inds)):
                labels_per_sp = im_target[l_inds[i]]
                u_labels_per_sp = np.unique(labels_per_sp)
                hist = np.zeros(len(u_labels_per_sp))
                for j in range(len(hist)):
                    hist[j] = len(
                        np.where(labels_per_sp == u_labels_per_sp[j])[0])
                im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
            target = torch.from_numpy(im_target)
            if use_cuda:
                target = target.cuda()
            target = Variable(target)
            loss = loss_fn(output, target)

        if version == "continuity":
            outputHP = output.reshape((im.shape[0], im.shape[1], nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)
            loss = stepsize_sim * \
                loss_fn(output, target) + stepsize_con * (lhpy + lhpz)

        loss.backward()
        optimizer.step()

        print(batch_idx, '/', maxIter, '|',
              ' label num :', nLabels, ' | loss :', loss.item())

        if batch_idx == maxIter - 1:
            final = im_target.reshape(im.shape[0:2])
            file_name = file.strip('.pngjp')
            final.tofile(("predictions/" + test_or_ct + "/" + version + isbatch(batch) + "/" +
                         file_name + ".csv"), sep=",")

        if nLabels <= minLabels:
            print("nLabels", nLabels, "reached minLabels", minLabels, ".")
            final = im_target.reshape(im.shape[0:2])
            file_name = file.strip('.pngjp')
            final.tofile(("predictions/" + test_or_ct + "/" + version + isbatch(batch) + "/" +
                         file_name + ".csv"), sep=",")
            break

    if test_or_ct == "CT":
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array(
            [label_colours[c % nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        cv2.imwrite(("single_predictions/continuity/" +
                    (file).strip('.pngjg') + ".png"), im_target_rgb)
