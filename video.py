import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

import pipeline as ld
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

calib = np.load("calib.npz")

img =[]

test_images=os.listdir("test_images/")

image=mpimg.imread("test_images/"+test_images[0])
lane_detector =  ld.lane_detection(image, calib['arr_0'], calib['arr_1'])
#vertices = np.float32([[[287,670],[603,444],[678,444],[1031,670]]])
vertices = np.float32([[[287,670],[583,458],[701,458],[1031,670]]])
destination = np.float32([[[300,700],[300,20],[980,20],[980,700]]])

n=8
# Calibrate the warp matrix for perspective transforms
lane_detector.calibrate_warp(vertices,destination)
lane_detector.reset_lanes()


output = 'harder_challenge_video_processed.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
clip = clip1.fl_image(lane_detector.image_pipeline)
clip.write_videofile(output, audio=False)
clip1.reader.close()
clip1.close()