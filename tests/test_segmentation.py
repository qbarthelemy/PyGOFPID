import cv2 as cv
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pygofpid.segmentation import FrameDifferencing, ViBe

np.random.seed(17)


###############################################################################


def test_framedifferencing():
    threshold = 7.5
    fd = FrameDifferencing(threshold=threshold)

    size = (2, 3)
    img = np.zeros(size, dtype=np.uint8)
    out = fd.apply(img)
    assert_array_equal(out, np.zeros(size))

    img = np.array([[6, 7, 8], [20, 10, 5]], dtype=np.uint8)
    out = fd.apply(img)
    assert_array_equal(out, np.array([[0, 0, 255], [255, 255, 0]]))


@pytest.mark.parametrize("frg_detect", [
    cv.createBackgroundSubtractorMOG2,
    cv.createBackgroundSubtractorKNN,
    ViBe,
    FrameDifferencing,
])
@pytest.mark.parametrize("size", [(32, 28), (28, 32, 1), (32, 28, 3)])
def test_segmentation(frg_detect, size):
    fd = frg_detect()

    for _ in range(50):
        img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
        out = fd.apply(img)
        assert out.shape == size[:2]
        assert out.dtype == np.uint8
        if frg_detect == FrameDifferencing and len(size) > 2:
            return
        assert np.all(np.isin(out, [0, 127, 255]))
