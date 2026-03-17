import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pygofpid.segmentation import FrameDifferencing, ViBe

np.random.seed(17)


###############################################################################


@pytest.mark.parametrize("size", [(64, 50), (50, 64, 1), (32, 28, 3)])
def test_framedifferencing(size):
    fd = FrameDifferencing(threshold=7.5)

    img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    out = fd.apply(img)
    assert_array_equal(out, np.zeros(size[:2]))

    for _ in range(3):
        img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
        out = fd.apply(img)
        assert out.shape == size[:2]


@pytest.mark.parametrize("size", [(32, 28), (28, 32, 1), (32, 28, 3)])
def test_vibe(size):
    vibe = ViBe()

    for _ in range(50):
        img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
        out = vibe.apply(img)
        assert out.shape == size[:2]
