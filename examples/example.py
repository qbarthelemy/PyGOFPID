"""
===============================================================================
Example of perimeter intrusion detection.
===============================================================================

Input video can be dowloaded here
https://pythonprogramming.net/static/images/opencv/people-walking.mp4
"""

import cv2 as cv
from pygofpid.gofpid import GOFPID


###############################################################################

# GOFPID with default parameters
gofpid = GOFPID().fit()

video_filename = 'people-walking.mp4'
capture = cv.VideoCapture(video_filename)

if not capture.isOpened():
    print('Unable to open input filemane')
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    y = gofpid.predict(frame)

    cv.imshow('Frame', frame)
    cv.imshow('Motion mask', gofpid.motion_mask_)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


###############################################################################
