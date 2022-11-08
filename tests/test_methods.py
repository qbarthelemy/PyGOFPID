"""Tests for module gofpid."""

import pytest
import numpy as np
import cv2 as cv
from pygofpid.methods import GOFPID

np.random.seed(17)


n_reps = 5

perimeter = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64)
perspective = np.array([[0.1, 0.5], [0.3, 0.9], [0.8, 0.1], [0.9, 0.25]])
post_filter = {
    'perimeter': perimeter,
    'anchor': 'center',
    'perspective': perspective,
    'perspective_coeff': 0.5,
    'presence_min': 3,
    'distance_min': 0.25,
}


def test_gofpid_errors():
    """Test GOFPID errors."""
    gofpid = GOFPID(post_filter=post_filter).initialize()

    img1 = np.random.randint(0, high=255, size=(64, 64, 3), dtype=np.uint8)
    gofpid.detect(img1)
    img2 = np.random.randint(0, high=255, size=(32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):  # input shape changed
        gofpid.detect(img2)


@pytest.mark.parametrize(
    "convert", [None, cv.COLOR_BGR2GRAY, cv.COLOR_RGB2GRAY]
)
def test_gofpid_convert(convert):
    """Test convert parameter."""
    gofpid = GOFPID(
        convert=convert,
        post_filter=post_filter,
    ).initialize()
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
    """Test blur parameters."""
    gofpid = GOFPID(
        blur=blur,
        post_filter=post_filter,
    ).initialize()
    img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    gofpid.detect(img)


def test_gofpid_blur_errors():
    """Test blur errors."""
    with pytest.raises(ValueError):  # no 'fun' in parameters
        GOFPID(
            blur={'ksize': (3, 3)},
            post_filter=post_filter,
        ).initialize()


@pytest.mark.parametrize("size", [(64, 64), (64, 64, 1), (64, 64, 3)])
@pytest.mark.parametrize("frg_detect", ['MOG2', 'KNN', 'FD'])
def test_gofpid_frgdetect(size, frg_detect):
    """Test frg_detect parameters."""
    gofpid = GOFPID(
        frg_detect=frg_detect,
        post_filter=post_filter,
    ).initialize()
    for _ in range(n_reps):
        img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
        gofpid.detect(img)


def test_gofpid_frgdetect_errors():
    """Test frg_detect errors."""
    with pytest.raises(ValueError):  # unknown method
        GOFPID(
            frg_detect='blabla',
            post_filter=post_filter,
        ).initialize()


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
    """Test mat_morph parameters."""
    gofpid = GOFPID(
        mat_morph=mat_morph,
        post_filter=post_filter,
    ).initialize()
    img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
    gofpid.detect(img)


def test_gofpid_matmorph_errors():
    """Test mat_morph errors."""
    with pytest.raises(ValueError):  # no 'fun' in parameters
        GOFPID(
            mat_morph=[
                {'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5))}
            ],
            post_filter=post_filter,
        ).initialize()


@pytest.mark.parametrize("anchor", ['center', 'bottom'])
def test_gofpid_postfilter(anchor):
    """Test postfilter parameters."""
    gofpid = GOFPID(
        post_filter={
            'perimeter': perimeter,
            'anchor': anchor,
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
    ).initialize()
    for _ in range(n_reps):
        img = np.random.randint(0, high=255, size=(64, 64), dtype=np.uint8)
        gofpid.detect(img)


@pytest.mark.parametrize(
    "post_filter",
    [
        {
            'anchor': 'center',
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        {
            'perimeter': np.array([[1, 5], [3, 9]]),
            'anchor': 'center',
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        {
            'perimeter': perimeter,
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        {
            'perimeter': perimeter,
            'anchor': 'blabla',
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        {
            'perimeter': perimeter,
            'anchor': 'center',
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        {
            'perimeter': perimeter,
            'anchor': 'center',
            'perspective': np.array([[1, 5, 2], [3, 9, 4]]),
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        {
            'perimeter': perimeter,
            'anchor': 'center',
            'perspective': perspective,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        {
            'perimeter': perimeter,
            'anchor': 'center',
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'distance_min': 0.25,
        },
        {
            'perimeter': perimeter,
            'anchor': 'center',
            'perspective': perspective,
            'perspective_coeff': 0.5,
            'presence_min': 3,
        },
    ]
)
def test_gofpid_postfilter_errors(post_filter):
    """Test post_filter errors."""
    with pytest.raises(ValueError):
        GOFPID(post_filter=post_filter).initialize()
