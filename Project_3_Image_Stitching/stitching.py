import cv2
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
project = "REMOTE"
project_folder = dname + '/' + project + '/'

# Load the images
image1 = cv2.imread(project_folder + '1.lower_half.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(project_folder + '1.upper_half.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the SIFT feature detector and extractor
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# # Draw keypoints on the images
# image1_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
# image2_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
matches_bf = bf.match(descriptors1, descriptors2)

# Sort the matches by distance (lower is better)
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

# Draw the top N matches
num_matches = 50
image_matches_bf = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_bf[:num_matches], None)

# Initialize the feature matcher using FLANN matching
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match the descriptors using FLANN matching
matches_flann = flann.match(descriptors1, descriptors2)

# Sort the matches by distance (lower is better)
matches_flann = sorted(matches_flann, key=lambda x: x.distance)

# Draw the top N matches
image_matches_flann = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_flann[:num_matches], None)

# Extract the matched keypoints
src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches_bf]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches_bf]).reshape(-1, 1, 2)

# Estimate the homography matrix using RANSAC
homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Print the estimated homography matrix
print("Estimated Homography Matrix:")
print(homography)


# # Display the images with keypoints
# cv2.imwrite(project_folder + '2.lower_half_sift.jpg', image1_keypoints)
# cv2.imwrite(project_folder + '2.upper_half_sift.jpg', image2_keypoints)
# # Display the images with matches
# cv2.imwrite(project_folder + '3.matches_brute-force.jpg', image_matches_bf)
# cv2.imwrite(project_folder + '3.matches_flann.jpg', image_matches_flann)