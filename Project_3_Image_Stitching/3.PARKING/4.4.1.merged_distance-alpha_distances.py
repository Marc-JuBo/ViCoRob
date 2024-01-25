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

white1 = np.ones(image1.shape[:2], np.uint8) * 255
white2 = np.ones(image2.shape[:2], np.uint8) * 255
white3 = np.ones(image3.shape[:2], np.uint8) * 255

translation = np.float32([[1, 0, 1600],[0, 1, 400],[0, 0, 1]])
size_x = image2.shape[1]+3200
size_y = image2.shape[0]+600
image1 = cv2.warpPerspective(image1, translation, (size_x, size_y))
image2 = cv2.warpPerspective(image2, translation, (size_x, size_y))
image3 = cv2.warpPerspective(image3, translation, (size_x, size_y))

white1 = cv2.warpPerspective(white1, translation, (size_x, size_y))
white2 = cv2.warpPerspective(white2, translation, (size_x, size_y))
white3 = cv2.warpPerspective(white3, translation, (size_x, size_y))

dist1 = cv2.distanceTransform(white1, cv2.DIST_L2, 3)
dist2 = cv2.distanceTransform(white2, cv2.DIST_L2, 3)
dist3 = cv2.distanceTransform(white3, cv2.DIST_L2, 3)
cv2.normalize(dist1, dist1, 0, 1.0, cv2.NORM_MINMAX)
cv2.normalize(dist2, dist2, 0, 1.0, cv2.NORM_MINMAX)
cv2.normalize(dist3, dist3, 0, 1.0, cv2.NORM_MINMAX)

h12 = np.load(project_folder + ".3.homography12.npy")
h32 = np.load(project_folder + ".3.homography32.npy")
# Warp the first (and third) image using the homography
image1 = cv2.warpPerspective(image1, h12, (size_x, size_y))
image3 = cv2.warpPerspective(image3, h32, (size_x, size_y))
dist1 = cv2.warpPerspective(dist1, h12, (size_x, size_y))
dist3 = cv2.warpPerspective(dist3, h32, (size_x, size_y))

for j in range(size_x): 
    for i in range(size_y):
        d1 = dist1[i][j]
        d2 = dist2[i][j]
        d3 = dist3[i][j]
        dist1[i][j] = 1 if ( d1 > 0 and d2 == 0 ) else d1
        dist1[i][j] = d1/(d1+d2) if (d1+d2)>0 else 0
        dist3[i][j] = 1 if ( d3 > 0 and d2 == 0 ) else d3
        dist3[i][j] = d3/(d3+d2) if (d3+d2)>0 else 0
        dist2[i][j] = 1 if ( d2 > 0 and d1 == 0 and d3 == 0 ) else 0
        dist2[i][j] = d2/(d1+d2) if ( d1 > 0 and d2 > 0 ) else dist2[i][j]
        dist2[i][j] = d2/(d3+d2) if ( d3 > 0 and d2 > 0 ) else dist2[i][j]



np.save(".4.4.d1.npy", dist1)
np.save(".4.4.d2.npy", dist2)
np.save(".4.4.d3.npy", dist3)




# # Display the images with keypoints
# cv2.imwrite(project_folder + '2.lower_half_sift.jpg', image1_keypoints)
# cv2.imwrite(project_folder + '2.upper_half_sift.jpg', image2_keypoints)
# # Display the images with matches
# cv2.imwrite(project_folder + '3.matches_brute-force.jpg', image_matches_bf)
# cv2.imwrite(project_folder + '3.matches_flann.jpg', image_matches_flann)