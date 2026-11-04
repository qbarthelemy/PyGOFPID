import cv2 as cv
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pygofpid.segmentation import (
    FrameDifferencing,
    ViBe,
    MultipleForegroundEstimators,
)

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


@pytest.mark.parametrize("frg_detector", [
    cv.createBackgroundSubtractorMOG2,
    cv.createBackgroundSubtractorKNN,
    ViBe,
    FrameDifferencing,
    MultipleForegroundEstimators,
])
@pytest.mark.parametrize("size", [(32, 28), (28, 32, 1), (32, 28, 3)])
def test_segmentation(frg_detector, size):
    fd = frg_detector()

    for _ in range(15):
        img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
        out = fd.apply(img)
        assert out.shape == size[:2]
        assert out.dtype == np.uint8
        if frg_detector == MultipleForegroundEstimators:
            return
        assert np.all(np.isin(out, [0, 85, 127, 170, 255]))


@pytest.mark.parametrize("frg_detector", [
    cv.createBackgroundSubtractorMOG2,
    cv.createBackgroundSubtractorKNN,
    ViBe,
    FrameDifferencing,
    MultipleForegroundEstimators,
])
@pytest.mark.parametrize("size", [(53, 67), (50, 64, 1), (48, 60, 3)])
def test_segmentation_video_intrusion(get_video, frg_detector, size):
    """Test segmentation on a video where a square crosses the frame."""
    frames, boxes = get_video(size=size)
    fd = frg_detector()

    box_prev = None
    for i, (frame, box) in enumerate(zip(frames, boxes)):
        fg = fd.apply(frame) > 0

        if i >= 5:  # let estimators learn background, KNN needs 4 frames
            # region where foreground is expected: current and previous
            # positions of square, since frame differencing detects both
            region = np.zeros(fg.shape, dtype=bool)
            for b in (box, box_prev):
                if b is not None:
                    region[b[1]:b[3], b[0]:b[2]] = True
            assert np.mean(fg[~region]) < 0.01, f'false detections, frame {i}'

            if box is not None and box_prev is not None:
                square = fg[box[1]:box[3], box[0]:box[2]]
                assert np.mean(square) > 0.1, f'square missed, frame {i}'

        box_prev = box


@pytest.mark.parametrize("comb_method", [np.median, np.mean])
@pytest.mark.parametrize("size", [(12, 28), (28, 16, 1), (12, 16, 3)])
def test_multiple_foreground_estimators(comb_method, size):
    fd = MultipleForegroundEstimators(
        frg_methods = [
            cv.createBackgroundSubtractorKNN(),
            ViBe(),
            FrameDifferencing()
        ],
        comb_method = comb_method,
    )

    for _ in range(10):
        img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
        fd.apply(img)
