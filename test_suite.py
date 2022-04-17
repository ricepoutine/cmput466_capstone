import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import genfromtxt
import cv2

np.random.seed(0)


def compute_iou(actual, pred):
    current = confusion_matrix(actual, pred, labels=range(1, max_label+1))
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

# 200 test images


# load ground truth
data_dir = pjoin(dirname(os.path.realpath(__file__)),
                 'data', 'groundTruth', 'test')

test_files = []
for f in os.listdir(data_dir):
    test_files.append(f)

# consistency of tests, debug
test_files.sort()
mat_fname = pjoin(data_dir, '2018.mat')

mat_contents = sio.loadmat(mat_fname)
# pick best ground truth segmentation
# number of possible true masks
num_truths = mat_contents['groundTruth'].shape[1]

groundTruth = mat_contents['groundTruth'][0, 0]
actual = groundTruth['Segmentation'][0, 0]
max_label = np.amax(actual)
im_size = actual.shape

# load predicted mask
pred = (genfromtxt('continuity.csv', delimiter=',')
        ).reshape(actual.shape).astype(np.int64)


pred = pred.flatten()
actual = actual.flatten()

# find most frequent label in pred and set equal to ground truth label for miou measure
for i in range(1, max_label+1):
    indices = np.where(actual == i)[0]
    pred_label = np.bincount(pred[indices]).argmax()
    pred[np.where(pred == pred_label)] = i

# calculate miou
miou = compute_iou(actual, pred)
print(miou)

actual.tofile("true_mask.csv", sep=",")
pred.tofile("pred_mask.csv", sep=',')

# view true mask
label_colours = np.random.randint(255, size=(max_label, 3))
im_true_rgb = np.array(
    [label_colours[c % max_label] for c in actual])
cv2.imwrite("true_mask.png", im_true_rgb.reshape(im_size[0], im_size[1], 3))
im_pred_rgb = np.array(
    [label_colours[c % max_label] for c in pred])
cv2.imwrite("pred_mask.png", im_pred_rgb.reshape(im_size[0], im_size[1], 3))
