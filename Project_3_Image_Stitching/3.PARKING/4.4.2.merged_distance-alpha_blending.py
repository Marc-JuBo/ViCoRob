import cv2
import os
import numpy as np
import imutils

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
image1 = cv2.warpPerspective(image1, h12, (size_x, size_y))
image3 = cv2.warpPerspective(image3, h32, (size_x, size_y))



dist1 = np.load(project_folder + ".4.4.d1.npy")
dist2 = np.load(project_folder + ".4.4.d2.npy")
dist3 = np.load(project_folder + ".4.4.d3.npy")



dist1 = cv2.cvtColor(dist1,cv2.COLOR_GRAY2RGB)
dist2 = cv2.cvtColor(dist2,cv2.COLOR_GRAY2RGB)
dist3 = cv2.cvtColor(dist3,cv2.COLOR_GRAY2RGB)


cv2.imwrite(project_folder + 'BRUT999.jpeg', image1)

cv2.imwrite(project_folder + '.4.4.d1.jpeg', dist1*255)
cv2.imwrite(project_folder + '.4.4.d2.jpeg', dist2*255)
cv2.imwrite(project_folder + '.4.4.d3.jpeg', dist3*255)

blended_dist_p1 = dist1 + dist2

# Stitching the warped images with the second image
blended_image_p1 = image1 * (1-dist2) + image2 * dist2
blended_image_final = image3 * dist3 + blended_image_p1 * 1 
# Notice whe don't extract anything else from blended_image_p1, cause it's already been extracted on line 52.

# Display the blended image
cv2.imwrite(project_folder + '4.4.merged_distance-alpha.jpeg', blended_image_final)



# # Display the images with keypoints
# cv2.imwrite(project_folder + '2.lower_half_sift.jpg', image1_keypoints)
# cv2.imwrite(project_folder + '2.upper_half_sift.jpg', image2_keypoints)
# # Display the images with matches
# cv2.imwrite(project_folder + '3.matches_brute-force.jpg', image_matches_bf)
# cv2.imwrite(project_folder + '3.matches_flann.jpg', image_matches_flann)