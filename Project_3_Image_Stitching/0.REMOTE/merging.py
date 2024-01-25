import cv2
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
project_folder = dname + '/'

# Load the images
image1 = cv2.imread(project_folder + '1.lower_half.jpg')
image2 = cv2.imread(project_folder + '1.upper_half.jpg')

translation = np.float32([[1, 0, 50],[0, 1, 50],[0, 0, 1]])
size_x = image2.shape[1]+160
size_y = image2.shape[0]+100
image1 = cv2.warpPerspective(image1, translation, (size_x, size_y))
image2 = cv2.warpPerspective(image2, translation, (size_x, size_y))

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher()

# Match the descriptors using brute-force matching
matches_bf = bf.match(descriptors1, descriptors2)

# Initialize the feature matcher using FLANN matching
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match the descriptors using FLANN matching
matches_flann = flann.match(descriptors1, descriptors2)

# Select the top N matches
# num_matches = 50
# matches_bf = sorted(matches_bf, key=lambda x: x.distance)[:num_matches]
# matches_flann = sorted(matches_flann, key=lambda x: x.distance)[:num_matches]

# Extract matching keypoints
src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches_flann]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches_flann]).reshape(-1, 1, 2)

# Estimate the homography matrix
homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 1.0)

homography_copy = np.array([

    [-1.05    ,  0.013  , 526],
    [-0.024   , -0.99   , 473],
    [-0.000046, -0.00024,   1]

    ])

homography_affine = np.array([

    [-1.05  ,  0.013 , 526],
    [-0.024 , -0.99  , 473],
    [ 0     ,  0     ,   1]

    ])

homography_linear = np.array([

    [-1.,  0 , 526],
    [ 0 , -1 , 473],
    [ 0 ,  0 ,   1]

    ])

homography_identity = np.array([

    [ 1.,  0 , 0],
    [ 0 ,  1 , 0],
    [ 0 ,  0 , 1]

    ])


# Warp the first image using the homography
result = cv2.warpPerspective(image1, homography, (size_x, size_y))
result_affine = cv2.warpPerspective(image1, homography_affine, (size_x, size_y))
result_linear = cv2.warpPerspective(image1, homography_linear, (size_x, size_y))
result_identity = cv2.warpPerspective(image1, homography_identity, (size_x, size_y))


# Display the blended image
cv2.imwrite(project_folder + '4.transformed_final.jpg', result)
cv2.imwrite(project_folder + '4.transformed_test_affine.jpg', result_affine)
cv2.imwrite(project_folder + '4.transformed_test_linear.jpg', result_linear)
cv2.imwrite(project_folder + '4.transformed_test_identity.jpg', result_identity)

cv2.imwrite(project_folder + '4.image2.jpg', image2)

# Blending the warped image with the second image using alpha blending
alpha = 0.5  # blending factor
blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)
blended_image_affine = cv2.addWeighted(result_affine, alpha, image2, 1 - alpha, 0)
blended_image_linear = cv2.addWeighted(result_linear, alpha, image2, 1 - alpha, 0)
blended_image_identity = cv2.addWeighted(result_identity, alpha, image2, 1 - alpha, 0)

# Display the blended image
cv2.imwrite(project_folder + '4.merged_final.jpg', blended_image)
cv2.imwrite(project_folder + '4.merged_test_affine.jpg', blended_image_affine)
cv2.imwrite(project_folder + '4.merged_test_linear.jpg', blended_image_linear)
cv2.imwrite(project_folder + '4.merged_test_identity.jpg', blended_image_identity)

# # Display the images with keypoints
# cv2.imwrite(project_folder + '2.lower_half_sift.jpg', image1_keypoints)
# cv2.imwrite(project_folder + '2.upper_half_sift.jpg', image2_keypoints)
# # Display the images with matches
# cv2.imwrite(project_folder + '3.matches_brute-force.jpg', image_matches_bf)
# cv2.imwrite(project_folder + '3.matches_flann.jpg', image_matches_flann)