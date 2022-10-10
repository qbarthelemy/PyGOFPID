"""Tests for module gofpid."""

import pytest
import numpy as np
import cv2 as cv
from pygofpid.gofpid import GOFPID, get_centers, get_bottoms

np.random.seed(17)

post_filter={
    'perimeter': np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64),
    'anchor': 'center',
    'perspective': np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64), #TODO
}

def test_gofpid_errors():
    """Test GOFPID errors."""
    gofpid = GOFPID(post_filter=post_filter).init()

    img1 = np.random.randint(0, high=255, size=(64, 64, 3), dtype=np.uint8)
    gofpid.detect(img1)
    img2 = np.random.randint(0, high=255, size=(32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):  # input shape changed
        gofpid.detect(img2)


@pytest.mark.parametrize("convert",
    [None, cv.COLOR_BGR2GRAY, cv.COLOR_RGB2GRAY]
)
def test_gofpid_convert(convert):
    """Test parameter convert."""
    gofpid = GOFPID(
        convert=convert,
        post_filter=post_filter,
    ).init()
    img = np.random.randint(0, high=255, size=(64, 64, 3), dtype=np.uint8)
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
    gofpid = GOFPID(
        blur=blur,
        post_filter=post_filter,
    ).init()
    img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    gofpid.detect(img)


def test_gofpid_blur_errors():
    """Test parameter blur errors."""
    with pytest.raises(ValueError):  # no 'fun' in parameters
        GOFPID(
            blur={'ksize': (3, 3)},
            post_filter=post_filter,
        ).init()


@pytest.mark.parametrize("size", [(64, 64), (64, 64, 1), (64, 64, 3)])
@pytest.mark.parametrize("frg_detect", ['MOG2', 'KNN', 'FD'])
def test_gofpid_frgdetect(size, frg_detect):
    """Test parameter frg_detect."""
    gofpid = GOFPID(
        frg_detect=frg_detect,
        post_filter=post_filter,
    ).init()
    img1 = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    gofpid.detect(img1)
    img2 = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    gofpid.detect(img2)


def test_gofpid_frgdetect_errors():
    """Test parameter frg_detect errors."""
    with pytest.raises(ValueError):  # unknown method
        GOFPID(
            frg_detect='blabla',
            post_filter=post_filter,
        ).init()


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
    gofpid = GOFPID(
        mat_morph=mat_morph,
        post_filter=post_filter,
    ).init()
    img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    gofpid.detect(img)


def test_gofpid_matmorph_errors():
    """Test parameter mat_morph errors."""
    with pytest.raises(ValueError):  # no 'fun' in parameters
        GOFPID(
            mat_morph=[
                {'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5))}
            ],
            post_filter=post_filter,
        ).init()

# TODO test_gofpid_postfilter():

def test_gofpid_intdetect_errors():
    """Test parameter int_detect errors."""
    with pytest.raises(ValueError):  # no 'presence_max' in parameters
        GOFPID(
            post_filter=post_filter,
            int_detect={'fake': 1},
        ).init()


def test_get_centers():
    """Test get_centers."""
    contours = [np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32)]
    centers = get_centers(contours)
    assert centers[0][0] == 5
    assert centers[0][1] == 5


def test_get_bottoms():
    """Test get_bottoms."""
    contours = [np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.int32)]
    bottoms = get_bottoms(contours)
    assert bottoms[0][0] == 5
    assert bottoms[0][1] == 10

