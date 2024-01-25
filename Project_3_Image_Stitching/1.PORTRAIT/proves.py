import cv2
import os
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
project_folder = dname + '/'

# Load the images
image1 = cv2.imread(project_folder + '1.right.png')
size_x = image1.shape[1]+800
size_y = image1.shape[0]+800


translation = np.float32([
    [1, 0, 400],
    [0, 1, 400],
    [0, 0, 1]])
image1_XL = cv2.warpPerspective(image1, translation, (size_x, size_y))
testing = np.float32([
    [ -1 , +0 , +0 ],
    [ +0 , -1 , +0 ],
    [ +0 , +0 , +1 ]])
image1_retocada = cv2.warpPerspective(image1, translation@testing, (size_x, size_y))


cv2.imwrite(project_folder + '99.proves_original.jpg', image1_XL)
cv2.imwrite(project_folder + '99.proves_retocada.jpg', image1_retocada)