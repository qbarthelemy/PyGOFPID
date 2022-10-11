"""Tests for module helpers."""

import pytest
from pytest import approx
import numpy as np

from pygofpid.helpers import (
    get_centers,
    get_bottoms,
    #is_in_rectangles,
    normalize_coords,
    unnormalize_coords,
)


np.random.seed(17)


def test_get_centers():
    """Test get_centers."""
    contours = [np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32)]
    center = get_centers(contours)[0]
    assert center[0] == 5
    assert center[1] == 5


def test_get_bottoms():
    """Test get_bottoms."""
    contours = [np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32)]
    bottom = get_bottoms(contours)[0]
    assert bottom[0] == 5
    assert bottom[1] == 10


@pytest.mark.parametrize(
    "coords, gt",
    [([32, 64], [0.25, 0.5]), ([16, 128], [0.125, 1]), ([128, 64],  [1, 0.5])],
)
def test_normalize_coords(coords, gt):
    ncoords = normalize_coords([coords], [128, 128])[0]
    assert gt == approx(ncoords)


@pytest.mark.parametrize(
    "gt, ncoords",
    [([32, 64], [0.25, 0.5]), ([16, 128], [0.125, 1]), ([128, 64],  [1, 0.5])],
)
def test_unnormalize_coords(gt, ncoords):
    coords = unnormalize_coords([ncoords], [128, 128])[0]
    assert gt == approx(coords)