import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import genfromtxt

np.random.seed(0)


def compute_iou(actual, pred):
    current = confusion_matrix(actual, pred, labels=range(1, max_label+1))
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


# load ground truths
data_dir = pjoin(dirname(os.path.realpath(__file__)),
                 'data', 'groundTruth', 'test')

ground_truth_files = []
for f in os.listdir(data_dir):
    ground_truth_files.append(f)

# order: continuity = 0, superpixel = 1, kmeans(#cluster = 2) = 2, kmeans(#clusters = 17) = 3
miou_list = [[], [], [], []]

for alg in range(4):
    if alg == 0:
        pred_dir = pjoin(dirname(os.path.realpath(__file__)),
                         'predictions', 'continuity')
    elif alg == 1:
        pred_dir = pjoin(dirname(os.path.realpath(__file__)),
                         'predictions', 'superpixel')
    elif alg == 2:
        pred_dir = pjoin(dirname(os.path.realpath(__file__)),
                         'predictions', 'kmeans3')
    else:
        pred_dir = pjoin(dirname(os.path.realpath(__file__)),
                         'predictions', 'kmeans15')

    for file in ground_truth_files:
        # load ground truth
        mat_fname = pjoin(data_dir, file)
        mat_contents = sio.loadmat(mat_fname)

        # dummy load to get size of image
        dummy = mat_contents['groundTruth'][0, 0]
        im = dummy['Segmentation'][0, 0]
        im_size = im.shape

        # load predicted mask
        maskID = file.strip('.mat')
        try:
            pred = (genfromtxt((pred_dir + '\\' + maskID + '.csv'), delimiter=',')
                    ).reshape(im_size).astype(np.int64)
        except:
            print("no prediction made on image id: ", maskID)
            continue

        # pick best ground truth segmentation
        # number of possible true masks
        num_truths = mat_contents['groundTruth'].shape[1]
        best_miou = 0
        for seg in range(num_truths):
            groundTruth = mat_contents['groundTruth'][0, seg]
            actual = groundTruth['Segmentation'][0, 0]
            max_label = np.amax(actual)

            pred = pred.flatten()
            actual = actual.flatten()

            # find most frequent label in pred and set equal to ground truth label for miou measure
            for i in range(1, max_label+1):
                indices = np.where(actual == i)[0]
                pred_label = np.bincount(pred[indices]).argmax()
                pred[np.where(pred == pred_label)] = i

            # calculate miou
            miou = compute_iou(actual, pred)
            print("mIOU: ", miou)

            if best_miou < miou:
                best_miou = miou

        miou_list[alg].append(best_miou)

np.array(miou_list, dtype=object).tofile("test_results.csv", sep=",")

"""
# save mask as csv
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
"""
