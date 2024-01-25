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

white1 = np.ones(image1.shape[:2], np.float32)
white2 = np.ones(image2.shape[:2], np.float32)
white3 = np.ones(image3.shape[:2], np.float32)

translation = np.float32([[1, 0, 1600],[0, 1, 400],[0, 0, 1]])
size_x = image2.shape[1]+3200
size_y = image2.shape[0]+600
image1 = cv2.warpPerspective(image1, translation, (size_x, size_y))
image2 = cv2.warpPerspective(image2, translation, (size_x, size_y))
image3 = cv2.warpPerspective(image3, translation, (size_x, size_y))

white1 = cv2.warpPerspective(white1, translation, (size_x, size_y))
white2 = cv2.warpPerspective(white2, translation, (size_x, size_y))
white3 = cv2.warpPerspective(white3, translation, (size_x, size_y))

h12 = np.load(project_folder + ".3.homography12.npy")
h32 = np.load(project_folder + ".3.homography32.npy")
# Warp the first (and third) image using the homography
image1 = cv2.warpPerspective(image1, h12, (size_x, size_y))
image3 = cv2.warpPerspective(image3, h32, (size_x, size_y))
white1 = cv2.warpPerspective(white1, h12, (size_x, size_y))
white3 = cv2.warpPerspective(white3, h32, (size_x, size_y))



for j in range(size_x): 
    for i in range(size_y):
        d1 = white1[i][j]
        d2 = white2[i][j]
        d3 = white3[i][j]
        if ( d1 > 0 and d2 > 0 ):
            white1[i][j] = 0.5
            white2[i][j] = 0.5
        if ( d3 > 0 and d2 > 0 ):
            white3[i][j] = 0.5
            white2[i][j] = 0.5





white1 = cv2.cvtColor(white1,cv2.COLOR_GRAY2RGB)
white2 = cv2.cvtColor(white2,cv2.COLOR_GRAY2RGB)
white3 = cv2.cvtColor(white3,cv2.COLOR_GRAY2RGB)


# Stitching the warped images with the second image
blended_image = image1 * white1 + image2 * white2 + image3 * white3

# Display the blended image
cv2.imwrite(project_folder + '4.2.merged_alpha_common-places.jpeg', blended_image)



# # Display the images with keypoints
# cv2.imwrite(project_folder + '2.lower_half_sift.jpg', image1_keypoints)
# cv2.imwrite(project_folder + '2.upper_half_sift.jpg', image2_keypoints)
# # Display the images with matches
# cv2.imwrite(project_folder + '3.matches_brute-force.jpg', image_matches_bf)
# cv2.imwrite(project_folder + '3.matches_flann.jpg', image_matches_flann)