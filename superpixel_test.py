#from __future__ import print_function
import os
from os.path import dirname, join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
from skimage import segmentation
import torch.nn.init

np.random.seed(0)

use_cuda = torch.cuda.is_available()

nChannel = 100
maxIter = 1000
minLabels = 3
lr = 0.1
nConv = 2
num_superpixels = 10000
compactness = 100
files = []

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
        x = self.bn3(x)
        return x


for file in files:
    # load image
    im = cv2.imread(pjoin(data_dir, file))
    data = torch.from_numpy(
        np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # slic
    labels = segmentation.slic(
        im, compactness=compactness, n_segments=num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    # train
    model = MyNet(data.size(1))
    if use_cuda:
        model.cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))
    for batch_idx in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        # superpixel refinement
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

        loss.backward()
        optimizer.step()

        print(batch_idx, '/', maxIter, '|',
              ' label num :', nLabels, ' | loss :', loss.item())

        if batch_idx == maxIter - 1:
            final = im_target.reshape(im.shape[0:2])
            file_name = file.strip('.jpg')
            final.tofile(("predictions/superpixel/" +
                          file_name + ".csv"), sep=",")

        if nLabels <= minLabels:
            print("nLabels", nLabels, "reached minLabels", minLabels, ".")
            final = im_target.reshape(im.shape[0:2])
            file_name = file.strip('.jpg')
            final.tofile(("predictions/superpixel/" +
                          file_name + ".csv"), sep=",")
            break


"""
output = model(data)[0]
output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
ignore, target = torch.max(output, 1)
im_target = target.data.cpu().numpy()
im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
cv2.imwrite("output_superpixel.png", im_target_rgb)
"""
