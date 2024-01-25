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

h12 = np.load(project_folder + ".3.homography12.npy")
h32 = np.load(project_folder + ".3.homography32.npy")
# Warp the first (and third) image using the homography
result12 = cv2.warpPerspective(image1, h12, (size_x, size_y))
result32 = cv2.warpPerspective(image3, h32, (size_x, size_y))


# Stitching the warped image with the second image using alpha blending
alpha = 0.50  # blending factor
merged_image_p1 = cv2.max(result12, image2)
merged_image_final = cv2.max(result32, merged_image_p1)

# Display the blended image
cv2.imwrite(project_folder + '4.merged_max.jpeg', merged_image_final)



# # Display the images with keypoints
# cv2.imwrite(project_folder + '2.lower_half_sift.jpg', image1_keypoints)
# cv2.imwrite(project_folder + '2.upper_half_sift.jpg', image2_keypoints)
# # Display the images with matches
# cv2.imwrite(project_folder + '3.matches_brute-force.jpg', image_matches_bf)
# cv2.imwrite(project_folder + '3.matches_flann.jpg', image_matches_flann)