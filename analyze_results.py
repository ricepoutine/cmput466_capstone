from os.path import dirname
import os
from numpy import genfromtxt
import numpy as np

# load test results
continuity_test_results = genfromtxt((dirname(os.path.realpath(__file__))+"/test_results_continuity.csv"), delimiter = ',')
superpixel_test_results = genfromtxt((dirname(os.path.realpath(__file__))+"/test_results_superpixel.csv"), delimiter = ',')
kmeans3_test_results = genfromtxt((dirname(os.path.realpath(__file__))+"/test_results_kmeans3.csv"), delimiter = ',')
kmeans15_test_results = genfromtxt((dirname(os.path.realpath(__file__))+"/test_results_kmeans15.csv"), delimiter = ',')
continuity_nobatch_test_results = genfromtxt((dirname(os.path.realpath(__file__))+"/test_results_continuity_nobatch.csv"), delimiter = ',')


print(np.mean(continuity_test_results), np.mean(superpixel_test_results), np.mean(kmeans3_test_results), np.mean(kmeans15_test_results), np.mean(continuity_nobatch_test_results))
