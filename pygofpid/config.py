"""Interactive windows to configure perimeter and perspective."""

import cv2 as cv
import numpy as np

from .helpers import (
    plot_lines,
    plot_rectangles,
    normalize_coords,
    unnormalize_coords,
)


def config_perimeter(
    img,
    perimeter,
    verbose=False,
    window_name="Configure perimeter",
):
    """Display window to configure perimeter.

    Press 'r' key to reset, 'c' to close.
    Left-click on a point and drag it to move it.
    Right-click on a line to add a new point, and on an existing point to
    remove it.

    Parameters
    ----------
    img : ndarray of int, shape (n_height, n_width) or \
            (n_height, n_width, n_channel)
        Frame displayed to configure perimeter.
    perimeter : ndarray, shape (n_points, 2)
        Initial perimeter, in normalized coordinates.
    verbose : bool, default=False
        Verbose mode.
    window_name : str, default="Configure perimeter"
        Name of window.

    Returns
    -------
    perimeter : ndarray, shape (n_points, 2)
        Configured perimeter, in normalized coordinates.
    """

    clone, thickness = img.copy(), _get_thickness(img)

    points = unnormalize_coords(perimeter, img.shape).tolist()
    i_point, i_line = -1, -1

    def move_line(event, x, y, flags, params):
        nonlocal i_point, i_line
        if event == cv.EVENT_LBUTTONDOWN:
            i_point = find_point((x, y), points, thickness)
        elif event == cv.EVENT_LBUTTONUP and i_point >= 0:
            points[i_point] = [x, y]
        elif event == cv.EVENT_RBUTTONDOWN:
            i_point = find_point((x, y), points, thickness)
            i_line = find_line((x, y), points, thickness)
        elif event == cv.EVENT_RBUTTONUP:
            if i_point >= 0:
                points.pop(i_point)
            elif i_line >= 0:
                points.insert(i_line + 1, [x, y])

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, move_line)
    while True:
        img = clone.copy()
        plot_lines(img, points, thickness)
        cv.imshow(window_name, img)
        key = cv.waitKey(1) & 0xFF
        if key == ord("r"):  # 'r' key => reset window
            img = clone.copy()
            points = unnormalize_coords(
                np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
                img.shape,
            ).tolist()
        elif key == ord("c"):  # 'c' key => close
            cv.destroyWindow(window_name)
            break

    perimeter = normalize_coords(points, img.shape)
    if verbose:
        print("Config perimeter:\n", perimeter)
    return perimeter


def config_perspective(
    img,
    perspective,
    verbose=False,
    window_name="Configure perspective",
):
    """Display window to configure perspective.

    Press 'c' key to close.
    Left-click on a corner of a rectangle and drag it to move it.

    Parameters
    ----------
    img : ndarray of int, shape (n_height, n_width) or \
            (n_height, n_width, n_channel)
        Frame displayed to configure perspective.
    perspective : ndarray, shape (4, 2)
        Initial perspective, in normalized coordinates.
    verbose : bool, default=False
        Verbose mode.
    window_name : str, default="Configure perspective"
        Name of window.

    Returns
    -------
    perspective : ndarray, shape (4, 2)
        Configured perspective, in normalized coordinates.
    """

    clone, thickness = img.copy(), _get_thickness(img)

    points = unnormalize_coords(perspective, img.shape, dtype=np.int32)
    i_point = -1

    def move_rectangle(event, x, y, flags, params):
        nonlocal i_point
        if event == cv.EVENT_LBUTTONDOWN:
            i_point = find_point((x, y), points, thickness)
        elif event == cv.EVENT_LBUTTONUP and i_point >= 0:
            points[i_point] = [x, y]

    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, move_rectangle)
    while True:
        img = clone.copy()
        plot_rectangles(img, points, thickness)
        cv.imshow(window_name, img)
        key = cv.waitKey(1) & 0xFF
        if key == ord("c"):  # 'c' key => close
            cv.destroyWindow(window_name)
            break

    perspective = normalize_coords(points, img.shape)
    if verbose:
        print("Config perspective:\n", perspective)
    return perspective


def _get_thickness(img):
    """Compute thickness of points displayed on configuration windows."""
    return unnormalize_coords(
        np.array([[0.02, 0.02]]),
        img.shape,
        dtype=np.int32,
    )[0]


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
