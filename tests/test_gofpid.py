"""Tests for module gofpid."""

import pytest
import numpy as np
import cv2 as cv
from pygofpid.gofpid import GOFPID


@pytest.mark.parametrize("convert",
    [None, cv.COLOR_BGR2GRAY, cv.COLOR_RGB2GRAY]
)
def test_gofpid_convert(convert):
    """Test convert parameter."""
    img = np.random.randint(0, high=255, size=(64, 64, 3), dtype=np.uint8)

    gofpid = GOFPID(convert=convert).fit()
    gofpid.predict(img)


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
@pytest.mark.parametrize("frg_detect", ['MOG2', 'KNN', 'FD'])
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
def test_gofpid_allparams(size, blur, frg_detect, mat_morph):
    """Test parameters."""
    img1 = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    img2 = np.random.randint(0, high=255, size=size, dtype=np.uint8)

    gofpid = GOFPID(
        blur=blur,
        frg_detect=frg_detect,
        mat_morph=mat_morph,
    ).fit()
    gofpid.predict(img1)
    gofpid.predict(img2)
