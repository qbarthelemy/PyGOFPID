"""Helpers."""

import cv2 as cv
import numpy as np


def change_extension(filename_input, ext_new="png"):
    ext_old = filename_input.split('.')[-1]
    filename_output = filename_input.replace(ext_old, ext_new)
    return filename_output


def read_first_frame(video_filename):
    """Read first frame of a video."""

    vidcap = cv.VideoCapture(video_filename)
    if not vidcap.isOpened():
        raise ValueError("Unable to open video %s." % video_filename)
    _, frame = vidcap.read()
    vidcap.release()
    frame_filename = change_extension(video_filename)
    cv.imwrite(frame_filename, frame)
    return frame


def plot_lines(X, points, thickness, c=(0, 0, 255)):
    n_points = len(points)
    if n_points >= 2:
        for i in range(n_points):
            cv.line(X, points[i % n_points], points[(i+1) % n_points], c, 2)
    plot_squares(X, points, thickness, c=c)


def plot_rectangles(X, points, thickness, c=(0, 0, 255)):
    cv.rectangle(X, points[0], points[1], c, 2)
    cv.rectangle(X, points[2], points[3], c, 2)
    plot_squares(X, points, thickness, c=c)


def plot_squares(X, points, thickness, c=(0, 0, 255)):
    for point in (points):
        cv.rectangle(X, point - thickness, point + thickness, c, -1)


def find_point(coord, points, thickness):
    """Find the point selected by coordinates.

    Parameters
    ----------
    coord : ndarray, shape (2,)
        Input coordinates.
    points : ndarray, shape (n_points, 2)
        Points.
    thickness : ndarray, shape (2,)
        Thickness of points.

    Returns
    -------
    i : int
        Index of selected point.
    """

    for i, point in enumerate(points):
        if cv.pointPolygonTest(
            np.array([
                [point[0] - thickness[0], point[1] - thickness[1]],
                [point[0] - thickness[0], point[1] + thickness[1]],
                [point[0] + thickness[0], point[1] + thickness[1]],
                [point[0] + thickness[0], point[1] - thickness[1]],
            ]),
            coord,
            False,
        ) >= 0:
            return i
    else:
        return -1


def find_line(coord, points, thickness):
    """Find the line selected by coordinates.

    Parameters
    ----------
    coord : ndarray, shape (2,)
        Input coordinates.
    points : ndarray, shape (n_points, 2)
        Points defining lines.
    thickness : ndarray, shape (2,)
        Thickness of lines.

    Returns
    -------
    i : int
        Index of selected line.
    """

    n_points = len(points)
    if n_points < 2:
        return -1
    points = np.asarray(points)
    for i in range(n_points):
        middle = (points[i % n_points] + points[(i+1) % n_points]) / 2
        normal_vector = points[(i+1) % n_points] - points[i % n_points]
        normal_vector = normal_vector / cv.norm(normal_vector)
        if cv.pointPolygonTest(
            np.array(
                [
                    [points[i % n_points][0], points[i % n_points][1]],
                    [middle[0] - normal_vector[1] * thickness[0],
                     middle[1] + normal_vector[0] * thickness[1]],
                    [points[(i+1) % n_points][0], points[(i+1) % n_points][1]],
                    [middle[0] + normal_vector[1] * thickness[0],
                     middle[1] - normal_vector[0] * thickness[1]],
                ],
                dtype=np.int32,
            ),
            coord,
            False,
        ) >= 0:
            return i
    else:
        return -1


def find_contours(mask):
    """Find contours in a binary mask.

    Parameters
    ----------
    mask : ndarray of int, shape (n_height, n_width, n_color)
        Binary mask.
    """

    outs = cv.findContours(
        mask,
        mode=cv.RETR_EXTERNAL,
        method=cv.CHAIN_APPROX_NONE,
    )

    if cv.__version__ < "4.0.0":
        _, contours, _ = outs
    else:
        contours, _ = outs

    return contours


def get_bottom(contour, dtype=np.int16):
    """Compute middle-bottom point of a contour."""

    moments = cv.moments(contour)
    x = int(moments["m10"] / moments["m00"])
    y = max(contour[..., 1])  # downward axis => bottom = max
    return np.array([x, y], dtype=dtype)


def get_center(contour, dtype=np.int16):
    """Compute center of a contour."""

    moments = cv.moments(contour)
    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return np.array([x, y], dtype=dtype)


def normalize_coords(coords, shape, dtype=np.float64):
    """Transform image coords (x,y) into normalized coords in [0,1]."""

    ncoords = np.asarray(coords, dtype=dtype)
    ncoords[..., 0] /= shape[1]
    ncoords[..., 1] /= shape[0]
    return ncoords


def unnormalize_coords(ncoords, shape, dtype=np.uint16):
    """Transform normalized coords (x,y) in [0,1] into image coords."""

    coords = np.asarray(ncoords, dtype=np.float64)
    coords[..., 0] *= shape[1]
    coords[..., 1] *= shape[0]
    return coords.astype(dtype)


def cdist_euclidean(XA, XB):
    """NumPy implementation of SciPy's cdist.

    Replace scipy.spatial.distance.cdist(XA, XB, "euclidean").
    """
    XA, XB = np.asarray(XA), np.asarray(XB)
    mA, mB = len(XA), len(XB)
    dm = np.zeros((mA, mB))
    for i in range(mA):
        for j in range(mB):
            dm[i, j] = np.linalg.norm(XA[i] - XB[j])
    return dm


class SimpleLinearRegression():
    """Simple linear regression.

    Simple linear regression: y = a * x + b.
    """

    def __init__(self):
        self.coeff = None
        self.intercept = None

    def fit(self, x, y):
        """Fit linear model on two samples.

        Fit the parameters (a,b) of a simple linear regression between two
        samples, such as: y = a * x + b

        Parameters
        ----------
        x : ndarray, shape (2,)
            Training values.
        y : ndarray, shape (2,)
            Target values.

        Returns
        -------
        self : object
            Fitted instance.
        """

        if len(x) != 2:
            raise ValueError("Fit only two samples.")
        if len(x) != len(y):
            raise ValueError("Inputs must have the same size.")

        self.coeff = (y[1] - y[0]) / (x[1] - x[0])
        self.intercept = y[0] - self.coeff * x[0]
        return self

    def predict(self, x):
        """Predict using the linear model.

        Predict value y from input x using a simple linear regression:
        y = a * x + b.

        Parameters
        ----------
        x : ndarray, shape (n,)
            Input values.

        Returns
        -------
        y : ndarray, shape (n,)
            Predicted values.
        """

        return self.coeff * np.asarray(x) + self.intercept
