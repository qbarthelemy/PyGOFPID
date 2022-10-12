"""Helpers."""

import numpy as np
import cv2 as cv


def get_first_frame(video_filename):
    """Get first frame of video."""
    vidcap = cv.VideoCapture(video_filename)
    if not vidcap.isOpened():
        raise ValueError('Unable to open input filemane.')
    _, frame = vidcap.read()
    vidcap.release()
    return frame


def get_centers(contours, dtype=np.int16):
    """Compute centers of contours."""
    centers = []
    for contour in contours:
        moments = cv.moments(contour)
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        centers.append([x, y])
    return np.array(centers, dtype=dtype)


def get_bottoms(contours, dtype=np.int16):
    """Compute middle-bottom points of contours."""
    bottoms = []
    for contour in contours:
        moments = cv.moments(contour)
        x = int(moments["m10"] / moments["m00"])
        y = max(contour[:, 1])  # downward axis => bottom = max
        bottoms.append([x, y])
    return np.array(bottoms, dtype=dtype)


def is_in_rectangles(coord, centers, thickness):
    for i, center in enumerate(centers):
        if cv.pointPolygonTest(
            np.array([
                [center[0] - thickness[0], center[1] - thickness[1]],
                [center[0] - thickness[0], center[1] + thickness[1]],
                [center[0] + thickness[0], center[1] + thickness[1]],
                [center[0] + thickness[0], center[1] - thickness[1]],
            ]),
            coord,
            False,
        ) >= 0:
            return i
    else:
        return -1


def normalize_coords(coords, shape):
    coords = np.asarray(coords, dtype=np.float64)
    coords[:, 0] /= shape[1]
    coords[:, 1] /= shape[0]
    return coords


def unnormalize_coords(coords, shape, dtype=np.uint8):
    coords = np.asarray(coords, dtype=np.float64)
    coords[:, 0] *= shape[1]
    coords[:, 1] *= shape[0]
    return coords.astype(dtype)
