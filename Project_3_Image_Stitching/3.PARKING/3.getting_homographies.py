import cv2
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
project_folder = dname + '/'

# Load the images
image1 = cv2.imread(project_folder + '1.L.jpeg')
image2 = cv2.imread(project_folder + '1.M.jpeg')
image3 = cv2.imread(project_folder + '1.R.jpeg')

translation = np.float32([[1, 0, 1600],[0, 1, 400],[0, 0, 1]])
size_x = image2.shape[1]+3200
size_y = image2.shape[0]+600
image1 = cv2.warpPerspective(image1, translation, (size_x, size_y))
image2 = cv2.warpPerspective(image2, translation, (size_x, size_y))
image3 = cv2.warpPerspective(image3, translation, (size_x, size_y))

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
keypoints3, descriptors3 = sift.detectAndCompute(gray3, None)

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher()

# Match the descriptors using brute-force matching
matches12 = bf.match(descriptors1, descriptors2)
matches32 = bf.match(descriptors3, descriptors2)

# Select the top N matches
num_matches = 50
matches12 = sorted(matches12, key=lambda x: x.distance)[:num_matches]
matches32 = sorted(matches32, key=lambda x: x.distance)[:num_matches]


# Extract matching keypoints
src_points12 = np.float32([keypoints1[match.queryIdx].pt for match in matches12]).reshape(-1, 1, 2)
dst_points12 = np.float32([keypoints2[match.trainIdx].pt for match in matches12]).reshape(-1, 1, 2)
src_points32 = np.float32([keypoints3[match.queryIdx].pt for match in matches32]).reshape(-1, 1, 2)
dst_points32 = np.float32([keypoints2[match.trainIdx].pt for match in matches32]).reshape(-1, 1, 2)

# Estimate the homography matrixes
homography12, _ = cv2.findHomography(src_points12, dst_points12, cv2.RANSAC, 5.0)
homography32, _ = cv2.findHomography(src_points32, dst_points32, cv2.RANSAC, 5.0)

# Extract the homography matrix, so we don't have to repeat the calculations each time.
np.save(".3.homography12.npy", homography12)
np.save(".3.homography32.npy", homography32)
