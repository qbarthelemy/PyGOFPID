"""
===============================================================================
Perimeter intrusion detection on i-LIDS sterile zone dataset.
===============================================================================

1 - Download i-LIDS sterile zone dataset;
2 - Set dataset path;
3 - Run the script.
"""

import cv2 as cv
import numpy as np
from pygofpid.methods import GOFPID


###############################################################################

# display video during analysis
display = False

# i-LIDS dataset path
data_path = r'//TODO/2006_ILIDS/'

# videos in Disk_2-Testing
video_names = [
    'SZTEA101a', 'SZTEA101b',
    'SZTEA102a', 'SZTEA102b',
    'SZTEA103a',
    'SZTEA104a',
    'SZTEA105a',
    'SZTEA201a', 'SZTEA201b',
    'SZTEA202a', 'SZTEA202b',
    'SZTEA203a',
    'SZTEA204a',
    'SZTEN101a', 'SZTEN101b', 'SZTEN101c', 'SZTEN101d',
    'SZTEN102a', 'SZTEN102b', 'SZTEN102c', 'SZTEN102d',
    'SZTEN103a', 'SZTEN103b',
    'SZTEN201a', 'SZTEN201b', 'SZTEN201c', 'SZTEN201d', 'SZTEN201e',
    'SZTEN202a', 'SZTEN202b', 'SZTEN202c', 'SZTEN202d',
    'SZTEN203a',
]


###############################################################################

def get_gofpid(view):

    # Intrusion detection config by view
    if view == '1':
        perimeter = np.array([
            [1.   , 0.138],
            [0.6  , 0.138],
            [0.463, 0.338],
            [0.   , 0.7  ],
            [0.   , 1.   ],
            [1.   , 1.   ]
        ])
        perspective = np.array([
            [0.65138889, 0.56770833],
            [0.575     , 0.85243056],
            [0.68333333, 0.20833333],
            [0.72777778, 0.34548611]
        ])
    elif view == '2':
        perimeter = np.array([
            [1.   , 0.75 ],
            [0.628, 0.388],
            [0.487, 0.205],
            [0.487, 0.112],
            [0.432, 0.   ],
            [0.   , 0.   ],
            [0.   , 1.   ],
            [1.   , 1.   ]
        ])
        perspective = np.array([
            [0.49444444, 0.5625    ],
            [0.40555556, 0.88541667],
            [0.31388889, 0.06944444],
            [0.275     , 0.19965278]
        ])
    else:
        raise ValueError(f'Unknown view {view}.')

    # Intrusion detection config on training videos
    config_frame_filename = (
        data_path + '/Disk_1-Training/calibration/SZ' + view + '_far.tif'
    )
    config_frame = cv.imread(config_frame_filename, cv.IMREAD_GRAYSCALE)

    gofpid = GOFPID(
        post_filter={
            #'display_config': True,
            'perimeter': perimeter,
            'anchor': 'bottom',
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
            'config_frame': config_frame,
        },
        tracking={
            'factor': 1.5,
            'absence_max': 10
        },
        alarm_def={
            'keep_alarm': True,
            'duration_min': 15,
        },
        verbose=True
    ).initialize()

    return gofpid


###############################################################################

# loop on testing videos
for video_name in video_names:

    print(video_name)

    video_filename = data_path + '/Disk_2-Testing/video/' + video_name + '.mov'
    vidcap = cv.VideoCapture(video_filename)
    if not vidcap.isOpened():
        raise ValueError(f'Unable to open video file {video_filename}')

    gofpid = get_gofpid(video_name[5])

    while True:
        _, frame = vidcap.read()
        if frame is None:
            break

        detection = gofpid.detect(frame)

        if display:
            gofpid.display(frame)
            cv.imshow('Frame', frame)
            if cv.waitKey(1) & 0xFF == ord('c'):
                vidcap.release()
                cv.destroyAllWindows()
                break

    #TODO: compute evaluation using detection value

###############################################################################
