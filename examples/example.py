"""
===============================================================================
Example of perimeter intrusion detection.
===============================================================================

1 - Download video here
https://pythonprogramming.net/static/images/opencv/people-walking.mp4;
2 - Put it in the same folder;
3 - Run the script.
"""

import cv2 as cv
from pygofpid.gofpid import GOFPID


###############################################################################

# Video
video_filename = 'people-walking.mp4'
vidcap = cv.VideoCapture(video_filename)
if not vidcap.isOpened():
    print('Unable to open input filemane')
    exit(0)

# Pipeline GOFPID
gofpid = GOFPID(
    post_filter={
        'perimeter': None,
        'anchor': 'center',
        'perspective': None,
        'presence_max': 3,
        'video_filename': video_filename,
    },
    verbose=True
).init()
#gofpid = GOFPID(blur=None, mat_morph=None).init()

# Detection
while True:
    _, frame = vidcap.read()
    if frame is None:
        break

    # intrusion detection
    y = gofpid.detect(frame)
    gofpid.display(frame)
    #cv.imshow('Motion mask', gofpid.foreground_mask_)

    if cv.waitKey(1) & 0xFF == ord('c'):
        vidcap.release()
        cv.destroyAllWindows()
        break


###############################################################################
