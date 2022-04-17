import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
# https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/
# https://www.kdnuggets.com/2019/08/introduction-image-segmentation-k-means-clustering.html
parser = argparse.ArgumentParser(
    description='KMeans Unsupervised Segmentation')
parser.add_argument('--clusters', metavar='N', default=10, type=int,
                    help='number of clusters')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
args = parser.parse_args()

# Read in the image
image = cv2.imread(args.input)

# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)


# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1, 3))

# Convert to float type
pixel_vals = np.float32(pixel_vals)


# the below line of code defines the criteria for the algorithm to stop running,
# which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
# becomes 90%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.90)

# then perform k-means clustering wit h number of clusters defined as 3
# also random centres are initially choosed for k-means clustering
k = args.clusters
retval, labels, centers = cv2.kmeans(
    pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))

cv2.imwrite("k-means_output.png", segmented_image)
