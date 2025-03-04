"""
===============================================================================
Perimeter intrusion detection on a single video.
===============================================================================

1 - Download video 'people-walking' here
https://pythonprogramming.net/static/images/opencv/people-walking.mp4;
2 - Put it in the same folder;
3 - Run the script;
4 - Press the "C" key to valid configuration pages.
"""

import cv2 as cv
from pygofpid.methods import GOFPID


###############################################################################

# Video
video_filename = 'people-walking.mp4'
vidcap = cv.VideoCapture(video_filename)
if not vidcap.isOpened():
    print('Unable to open video file :', video_filename)
    exit(0)


# Pipeline GOFPID
gofpid = GOFPID(
    post_filter={
        'display_config': True,
        'perimeter': None,
        'anchor': 'bottom',
        'perspective': None,
        'perspective_coeff': 0.5,
        'presence_min': 3,
        'distance_min': 0.25,
        'config_video_filename': video_filename,
    },
    verbose=True,
).initialize()


# Intrusion detection
while True:
    _, frame = vidcap.read()
    if frame is None:
        break

    gofpid.detect(frame)
    gofpid.display(frame)
    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xFF == ord('c'):
        vidcap.release()
        cv.destroyAllWindows()
        break


###############################################################################
