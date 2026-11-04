import cv2 as cv
import numpy as np
import pytest

from pygofpid.segmentation import FrameDifferencing as FD
from pygofpid.pipeline import GOFPID

np.random.seed(17)


n_reps = 5

perimeter = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
perspective = np.array([[0.6, 0.5], [0.5, 0.8], [0.7, 0.2], [0.7, 0.3]])
post_filter = {
    'perimeter': perimeter,
    'anchor': 'center',
    'perspective': perspective,
    'perspective_coeff': 0.5,
    'presence_min': 3,
    'distance_min': 0.25,
}
alarm_def = {
    'keep_alarm': True,
    'duration_min': 10,
}


###############################################################################


def test_gofpid():
    gofpid = GOFPID().initialize()
    assert isinstance(gofpid, GOFPID)

    size = (64, 64, 3)
    for _ in range(10):
        img = np.random.randint(0, high=255, size=size, dtype=np.uint8)
        y = gofpid.detect(img)

    assert isinstance(y, bool)
    assert gofpid.foreground_mask_.shape == size[:2]
    assert isinstance(gofpid.blobs_, list)
    assert isinstance(gofpid.tracked_blobs_, list)


def test_gofpid_errors():
    """Test GOFPID errors."""
    gofpid = GOFPID(
        post_filter=post_filter,
    ).initialize()

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


@pytest.mark.parametrize("size", [(64, 50), (64, 32, 1), (32, 20, 3)])
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
    with pytest.raises(KeyError):  # no 'fun' in parameters
        GOFPID(
            blur={'ksize': (3, 3)},
            post_filter=post_filter,
        ).initialize()


@pytest.mark.parametrize("size", [(32, 20), (20, 32, 1), (32, 20, 3)])
@pytest.mark.parametrize("frg_detect", [
    'MOG2', 'KNN', 'ViBe', 'FD', ['MOG2', 'FD'], FD,
])
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


@pytest.mark.parametrize("size", [(64, 50), (20, 32, 1), (32, 20, 3)])
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
    with pytest.raises(KeyError):  # no 'fun' in parameters
        GOFPID(
            mat_morph=[
                {'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5))}
            ],
            post_filter=post_filter,
        ).initialize()


def test_gofpid_tracking_errors():
    """Test tracking errors."""
    with pytest.raises(KeyError):  # no 'factor' in parameters
        GOFPID(
            tracking={'blabla': 42},
            post_filter=post_filter,
        ).initialize()


@pytest.mark.parametrize("anchor", ['center', 'bottom'])
def test_gofpid_postfilter(anchor):
    """Test post_filter parameters."""
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
        img = np.random.randint(0, high=255, size=(64, 50), dtype=np.uint8)
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
    with pytest.raises((KeyError, ValueError)):
        GOFPID(
            post_filter=post_filter,
        ).initialize()


@pytest.mark.parametrize("keep_alarm", [True, False])
@pytest.mark.parametrize("duration_min", [2, 3, 4])
def test_gofpid_alarmdef(keep_alarm, duration_min):
    """Test alarm_def parameters."""
    gofpid = GOFPID(
        alarm_def={
            'keep_alarm': keep_alarm,
            'duration_min': duration_min,
        },
    ).initialize()

    for _ in range(n_reps):
        img = np.random.randint(0, high=255, size=(64, 50), dtype=np.uint8)
        gofpid.detect(img)


###############################################################################


# perspective where a person has the size of the square, whatever its depth
post_filter_square = {
    'perimeter': perimeter,
    'anchor': 'center',
    'perspective': np.array(
        [[0.1, 0.3], [0.25, 0.55], [0.6, 0.5], [0.75, 0.75]]
    ),
    'perspective_coeff': 0.5,
    'presence_min': 3,
    'distance_min': 0.25,
}


@pytest.mark.parametrize("frg_detect", ['MOG2', 'KNN', 'ViBe'])
def test_gofpid_video_no_intrusion(get_video, frg_detect):
    """Test no alarm is raised on a video without intrusion."""
    frames, _ = get_video(with_square=False)
    gofpid = GOFPID(
        frg_detect=frg_detect,
        post_filter=post_filter_square,
    ).initialize()

    assert not any(gofpid.detect(frame) for frame in frames)


@pytest.mark.parametrize("frg_detect", ['MOG2', 'KNN', 'ViBe'])
@pytest.mark.parametrize("size", [(53, 67), (50, 64, 1), (48, 60, 3)])
def test_gofpid_video_intrusion(get_video, frg_detect, size):
    """Test intrusion detection on a video where a square crosses the frame."""
    frames, boxes = get_video(size=size)
    gofpid = GOFPID(
        frg_detect=frg_detect,
        post_filter=post_filter_square,
    ).initialize()

    y = [gofpid.detect(frame) for frame in frames]

    assert not any(y_ for y_, box in zip(y, boxes) if box is None), \
        'alarm raised without intrusion'
    assert any(y), 'intrusion not detected'
    i_first_square = next(i for i, box in enumerate(boxes) if box is not None)
    assert y.index(True) <= i_first_square + 5, 'intrusion detected too late'


def test_gofpid_square_perimeter(get_video):
    """Test alarm is raised only when the square enters the perimeter."""
    n_width = 64
    frames, boxes = get_video(size=(48, n_width))
    post_filter = post_filter_square.copy()
    post_filter['perimeter'] = np.array(
        [[0.5, 0], [0.5, 1], [1, 1], [1, 0]]  # right half of frame
    )
    gofpid = GOFPID(post_filter=post_filter).initialize()

    y = [gofpid.detect(frame) for frame in frames]

    assert any(y), 'intrusion not detected'
    box = boxes[y.index(True)]
    center_x = (box[0] + box[2]) / 2
    assert center_x >= n_width / 2, 'alarm raised outside perimeter'
