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
from pygofpid.methods import GOFPID


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
        'anchor': 'bottom',
        'perspective': None,
        'perspective_coeff': 0.5,
        'presence_max': 3,
        'video_filename': video_filename,
    },
    verbose=True
).init()

# Intrusion detection
while True:
    _, frame = vidcap.read()
    if frame is None:
        break

    y = gofpid.detect(frame)
    gofpid.display(frame)

    if cv.waitKey(1) & 0xFF == ord('c'):
        vidcap.release()
        cv.destroyAllWindows()
        break


###############################################################################
