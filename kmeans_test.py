import numpy as np
import cv2
import os
from os.path import dirname, join as pjoin

# cluster number = 3, 15
clusters = [3, 15]
files = []

# load test images
data_dir = pjoin(dirname(os.path.realpath(__file__)),
                 'data', 'images', 'test')
for f in os.listdir(data_dir):
    files.append(f)

for k in clusters:
    for file in files:
        # Prepare image
        image = cv2.imread(pjoin(data_dir, file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixel_vals = image.reshape((-1, 3))
        pixel_vals = np.float32(pixel_vals)

        # stop criteria: 100 iterations are run or the epsilon (which is the required accuracy) becomes 90%
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.90)

        # k-means clustering with variable number of clusters (random centres initially)
        retval, labels, centers = cv2.kmeans(
            pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # round predicted labels
        centers = np.uint8(centers)

        # reshape data into the original image dimensions
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((image.shape))

        # extract predicted mask by taking only 1 color value from image reconstruction
        file_name = file.strip('.jpg')
        segmented_image[:, :, 0].tofile(
            ("predictions/kmeans"+str(k)+"/" + file_name + ".csv"), sep=",")

        # view image
        cv2.imwrite("k-means_output.png", segmented_image)
