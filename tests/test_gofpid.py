"""Tests for module gofpid."""

import pytest
import numpy as np
import cv2 as cv
from pygofpid.gofpid import GOFPID

np.random.seed(17)


@pytest.mark.parametrize("convert",
    [None, cv.COLOR_BGR2GRAY, cv.COLOR_RGB2GRAY]
)
def test_gofpid_convert(convert):
    """Test parameter convert."""
    img = np.random.randint(0, high=255, size=(64, 64, 3), dtype=np.uint8)

    gofpid = GOFPID(convert=convert).init()
    gofpid.detect(img)


@pytest.mark.parametrize("size", [(64, 64), (64, 64, 1), (64, 64, 3)])
@pytest.mark.parametrize(
    "blur",
    [
        None,
        {
            'fun': cv.GaussianBlur,
            'ksize': (3, 3),
            'borderType': cv.BORDER_DEFAULT,
        },
        {
            'fun': cv.blur,
            'borderType': cv.BORDER_REPLICATE,
        },
        {
            'fun': cv.blur,
            'ksize': (3, 3),
        },
    ]
)
def test_gofpid_blur(size, blur):
    """Test parameter blur."""
    img = np.random.randint(0, high=255, size=size, dtype=np.uint8)

    gofpid = GOFPID(blur=blur).init()
    gofpid.detect(img)


@pytest.mark.parametrize("size", [(64, 64), (64, 64, 1), (64, 64, 3)])
@pytest.mark.parametrize("frg_detect", ['MOG2', 'KNN', 'FD'])
def test_gofpid_frgdetect(size, frg_detect):
    """Test parameter frg_detect."""
    img1 = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    img2 = np.random.randint(0, high=255, size=size, dtype=np.uint8)

    gofpid = GOFPID(frg_detect=frg_detect).init()
    gofpid.detect(img1)
    gofpid.detect(img2)


@pytest.mark.parametrize("size", [(64, 64), (64, 64, 1), (64, 64, 3)])
@pytest.mark.parametrize(
    "mat_morph",
    [
        None,
        [
            {
                'fun': cv.erode,
                'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5)),
            },
            {
                'fun': cv.dilate,
                'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5)),
            }
        ],
    ]
)
def test_gofpid_matmorph(size, mat_morph):
    """Test parameter mat_morph."""
    img = np.random.randint(0, high=255, size=size, dtype=np.uint8)

    gofpid = GOFPID(mat_morph=mat_morph).init()
    gofpid.detect(img)
