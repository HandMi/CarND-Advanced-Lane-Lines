import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

import pipeline as ld


calib = np.load("calib.npz")

img =[]

test_images=os.listdir("test_images/")

image=mpimg.imread("test_images/"+test_images[0])
lane_detector =  ld.lane_detection(image, calib['arr_0'], calib['arr_1'])
vertices = np.float32([[[287,670],[603,444],[678,444],[1031,670]]])
destination = np.float32([[[300,700],[300,20],[980,20],[980,700]]])

n=8
lane_detector.calibrate_warp(vertices,destination)
titles = ["Original Image", "Undistorted Image"]
fig, ax = plt.subplots(n,2,figsize=(50,100))
fig.tight_layout()

for i in range(0, n):
    lane_detector.reset_lanes()
    image=mpimg.imread("test_images/"+test_images[i])
    result = lane_detector.image_pipeline(image)
    ax[i][0].imshow(image)
    ax[i][0].set_title(titles[0])
    ax[i][0].axis('off')
    ax[i][1].imshow(result)
    ax[i][1].set_title(titles[1])
    ax[i][1].axis('off')

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.tight_layout()
plt.savefig('test.png')