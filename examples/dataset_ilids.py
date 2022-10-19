"""
===============================================================================
Perimeter intrusion detection on i-LIDS dataset.
===============================================================================

1 - Download i-LIDS dataset;
2 - Set paths and filenames;
3 - Run the script.
"""

import numpy as np
import cv2 as cv
from pygofpid.methods import GOFPID


###############################################################################

# Dataset
frame_filename = 'TODO' + '/Disk_1-Training/calibration/SZ2_far.tif'
frame = cv.imread(frame_filename, cv.IMREAD_GRAYSCALE)

is_view_1 = False

if is_view_1:
    perimeter = np.array(
        [[1.0       , 0.1145833  ],
         [0.6       , 0.11458333],
         [0.4625    , 0.30729167],
         [0.014     , 0.640625],
         [0.014     , 1.0       ],
         [1.0       , 1.0       ]]
    )
    perspective = np.array(
        [[0.65138889, 0.56770833],
         [0.575     , 0.85243056],
         [0.68333333, 0.20833333],
         [0.72777778, 0.34548611]]
    )
else:
    perimeter = np.array(
        [[0.98611111, 0.69791667],
         [0.62777778, 0.35590278],
         [0.50555556, 0.17534722],
         [0.50277778, 0.109375  ],
         [0.43333333, 0.01388889],
         [0.00416667, 0.00520833],
         [0.00694444, 0.99826389],
         [1.        , 1.        ]]
    )
    perspective = np.array(
        [[0.49444444, 0.5625    ],
         [0.40555556, 0.88541667],
         [0.31388889, 0.06944444],
         [0.275     , 0.19965278]]
    )

# Pipeline GOFPID
gofpid = GOFPID(
    post_filter={
        'display_config': True,
        'perimeter': perimeter,
        'anchor': 'bottom',
        'perspective': perspective,
        'perspective_coeff': 0.5,
        'presence_min': 3,
        'distance_min': 0.25,
        'config_frame': frame,
    },
    verbose=True
).initialize()

# Intrusion detection
# TODO

###############################################################################
