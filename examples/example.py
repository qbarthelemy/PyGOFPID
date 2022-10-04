"""
===============================================================================
Example of perimeter intrusion detection.
===============================================================================

First, download video here
https://pythonprogramming.net/static/images/opencv/people-walking.mp4
and put it in the same folder. Then, run the script.
"""

import cv2 as cv
from pygofpid.gofpid import GOFPID


###############################################################################

# GOFPID with default parameters
gofpid = GOFPID().fit()
#gofpid = GOFPID(blur=None, mat_morph=None).fit()

video_filename = 'people-walking.mp4'
capture = cv.VideoCapture(video_filename)

if not capture.isOpened():
    print('Unable to open input filemane')
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # intrusion detection
    y = gofpid.predict(frame)
    gofpid.display(frame)
    #cv.imshow('Motion mask', gofpid.foreground_mask_)

    if cv.waitKey(1) & 0xFF == ord('q'):
        capture.release()
        cv.destroyAllWindows()
        break


###############################################################################
