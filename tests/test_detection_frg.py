"""Tests for module gofpid."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from pygofpid.detection_frg import FrameDifferencing

np.random.seed(17)


###############################################################################


@pytest.mark.parametrize("threshold", [5, 10, 15])
def test_framedifferencing(threshold):
    size = (64, 64)
    img_in = np.random.randint(0, high=255, size=(*size, 3), dtype=np.uint8)

    fd = FrameDifferencing(threshold=threshold)
    img_out = fd.apply(img_in)
    assert_array_equal(img_out, np.zeros(size))

    img_out = fd.apply(img_in)
    assert img_out.shape == size
