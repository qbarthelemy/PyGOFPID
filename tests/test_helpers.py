"""Tests for module helpers."""

import pytest
from pytest import approx
import numpy as np

from pygofpid.helpers import (
    plot_lines,
    plot_rectangles,
    plot_squares,
    find_point,
    find_line,
    find_contours,
    get_bottom,
    get_center,
    normalize_coords,
    unnormalize_coords,
    cdist_euclidean,
    SimpleLinearRegression,
)


np.random.seed(17)


@pytest.mark.parametrize("fun", [plot_lines, plot_rectangles, plot_squares])
def test_plots(fun):
    """Test plots."""
    X = np.ones((100, 150, 3), dtype=np.uint8)
    points = np.array([[10, 40], [20, 80], [80, 20], [85, 30]])
    thickness = np.array([5, 5])
    fun(X, points, thickness)


@pytest.mark.parametrize(
    "coord, gt",
    [([11, 51], 0), ([13, 53], -1), ([87, 33], 3), ([128, 64], -1)],
)
def test_find_point(coord, gt):
    """Test find_point."""
    points = [[10, 50], [20, 90], [80, 20], [85, 35]]
    thickness = [2, 2]
    assert find_point(coord, points, thickness) == gt


@pytest.mark.parametrize(
    "coord, gt",
    [([10, 30], 0), ([11, 30], 0), ([13, 30], -1), ([50, 30], 2)],
)
def test_find_line(coord, gt):
    """Test find_line."""
    points = [[10, 10], [10, 50], [50, 50], [50, 10]]
    thickness = [2, 2]
    assert find_line(coord, points, thickness) == gt


def test_find_contours():
    """Test find_contours."""
    mask = np.random.randint(low=0, high=2, size=(64, 64), dtype=np.uint8)
    find_contours(mask)


def test_get_bottom():
    """Test get_bottom."""
    contour = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32)
    bottom = get_bottom(contour)
    assert bottom[0] == 5   # middle
    assert bottom[1] == 10  # bottom


def test_get_center():
    """Test get_center."""
    contour = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32)
    center = get_center(contour)
    assert center[0] == 5
    assert center[1] == 5


@pytest.mark.parametrize(
    "coords, gt",
    [([32, 64], [0.25, 0.5]), ([16, 128], [0.125, 1]), ([128, 64], [1, 0.5])],
)
def test_normalize_coords(coords, gt):
    """Test normalize_coords."""
    ncoords = normalize_coords([coords], [128, 128])[0]
    assert gt == approx(ncoords)


@pytest.mark.parametrize(
    "gt, ncoords",
    [([32, 64], [0.25, 0.5]), ([16, 128], [0.125, 1]), ([128, 64], [1, 0.5])],
)
def test_unnormalize_coords(gt, ncoords):
    """Test unnormalize_coords."""
    coords = unnormalize_coords([ncoords], [128, 128])[0]
    assert gt == approx(coords)


def test_cdist_euclidean():
    """Test cdist_euclidean."""
    XA = np.array([[0, 0], [0, 10]])
    XB = np.array([[0, 0], [0, 10], [10, 0]])
    dists = cdist_euclidean(XA, XB)
    assert dists.shape == (len(XA), len(XB))
    np.testing.assert_almost_equal(dists, [[0, 10, 10], [10, 0, 14.14213562]])


def test_simplelinearregression():
    """Test SimpleLinearRegression."""
    slr = SimpleLinearRegression()
    x, y = [3, 4], [38, 37]
    slr.fit(x, y)
    assert slr.predict([5]) == 36.0


def test_simplelinearregression_errors():
    """Test SimpleLinearRegression errors."""
    slr = SimpleLinearRegression()
    x = [-1, 0, 1, 2, 3, 4]
    y = [42, 41, 40, 39, 38, 37]
    with pytest.raises(ValueError):
        slr.fit(x, y)
    with pytest.raises(ValueError):
        slr.fit(x, y[1:])
