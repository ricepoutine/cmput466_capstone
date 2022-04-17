import numpy as np
import argparse
import cv2

# cluster number = 3, 15
parser = argparse.ArgumentParser(
    description='KMeans Unsupervised Segmentation')
parser.add_argument('--clusters', metavar='N', default=10, type=int,
                    help='number of clusters')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
args = parser.parse_args()

# Load image in correct format
image = cv2.imread(args.input)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshaping image into 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)


# stop criteria: 100 iterations are run or the epsilon (which is the required accuracy) becomes 90%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.90)

# k-means clustering with variable number of clusters (random centres initially)
k = args.clusters
retval, labels, centers = cv2.kmeans(
    pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # round predicted labels and reshape data into the original image dimensions
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((image.shape))

cv2.imwrite("k-means_output.png", segmented_image)
