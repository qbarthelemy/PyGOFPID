import pytest

from pygofpid.config import (
    find_point,
    find_line,
)


###############################################################################


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
