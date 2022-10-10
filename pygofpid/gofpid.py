
import numpy as np
from scipy.spatial.distance import cdist
import cv2 as cv


class GOFPID():
    """GOFPID: good old fashioned perimeter intrusion detection system.

    Using OpenCV, this class allows to build a pipeline using good old
    fashioned (GOF) computer vision methods for perimeter intrusion detection
    (PID):

    1. input frame denoising by spatial blurring;
    2. foreground detection by background subtraction;
    3. foreground mask denoising by mathematical morphology;
    4. foreground blob creation;
    5. blob tracking (WIP);
    6. post-filtering (perimeter, perspective) (TODO);
    7. intrusion detection.

    Parameters
    ----------
    convert : OpenCV color conversion codes | None, default=None
        Convert input frame from color to gray [Color]_.

    blur : OpenCV bur filter | None, default={'fun': cv.GaussianBlur, \
            'ksize': (3, 3), 'borderType': cv.BORDER_DEFAULT}
        OpenCV filter for spatial blurring and its parameters:

        - cv.GaussianBlur for a Gaussian filter [GaussBlur]_;
        - cv.blur for a normalized box filter [BoxBlur]_;
        - None, no processing.

    frg_detect : {'MOG2', 'KNN', 'FD'}, default='MOG2'
        Method for foreground detection [BkgSub]_:

        - 'MOG2' background subtraction by mixture of Gaussians [MOG2]_;
        - 'KNN' background subtraction by K-nearest neigbours [KNN]_;
        - 'FD' frame differencing.

    mat_morph : list of dict | None, default=[ \
            {'fun': cv.erode, 'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5))}, \
            {'fun': cv.dilate, 'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5))}]
        List of dictionaries containing OpenCV operators for mathematical
        morphology, and their parameters: 'cv.erode' for erosion [Ersn]_, and
        'cv.dilate' for dilation [Dltn]_.
        If None, no processing.

    post_filter : dict, default={'perimeter': None, 'anchor_point': 'center', \
            'perspective': None}
        Dictionary containing parameters to filter non-intrusions.
        perimeter: list of points. If None, a window allow to draw it.
        anchor_point: 'center' or 'bottom'.
        perspective: list of four points defining the minimum sizes of objects
        to detect. If None, a window allow to draw them.

    int_detect : dict, default={'presence_max': 3}
        Dictionary containing parameters to detect intrusion.
        presence_max: number of frames where objet is present and tracked
        before raising intrusion alarm.

    Attributes
    ----------
    foreground_mask_ : ndarray of int, shape (n_height, n_width, n_color)
        Foreground mask.

    blobs_ : list of n_blobs lists of n_points list of two int
        Instantaneous blobs created from foreground mask.

    tracked_blobs_ : tuple containing blobs, and two lists of int
        Tracked blobs created from instantaneous blobs: blobs, presence counts,
        abscence counts.

    References
    ----------
    .. [Color] https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    .. [GaussBlur] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    .. [BoxBlur] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37
    .. [BkgSub] https://docs.opencv.org/3.4.0/d7/df6/classcv_1_1BackgroundSubtractor.html
    .. [MOG2] https://docs.opencv.org/3.4.0/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    .. [KNN] https://docs.opencv.org/3.4.0/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    .. [Ersn] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
    .. [Dltn] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
    """  # noqa
    def __init__(
        self,
        convert=None,
        blur={
            'fun': cv.GaussianBlur,
            'ksize': (3, 3),
            'borderType': cv.BORDER_DEFAULT,
        },
        frg_detect='MOG2',
        mat_morph=[
            {
                'fun': cv.erode,
                'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5)),
            },
            {
                'fun': cv.dilate,
                'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5)),
            }
        ],
        post_filter={
            'perimeter': None,
            'anchor_point': 'center',
            'perspective': None,
        },
        int_detect={'presence_max': 3}
    ):
        self.convert = convert
        self.blur = blur
        self.frg_detect = frg_detect
        self.mat_morph = mat_morph
        self.post_filter = post_filter
        self.int_detect = int_detect

    def init(self):
        """Initialize, checking parameters and setting pipeline. No training.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.blur:
            if 'fun' not in self.blur.keys():
                raise ValueError('Parameter blur has no key "fun".')
            if 'ksize' not in self.blur.keys():
                self.blur['ksize'] = (3, 3)
            if 'borderType' not in self.blur.keys():
                self.blur['borderType'] = cv.BORDER_DEFAULT

        if self.frg_detect == 'MOG2':
            self._frg_detect_mth = cv.createBackgroundSubtractorMOG2()
        elif self.frg_detect == 'KNN':
            self._frg_detect_mth = cv.createBackgroundSubtractorKNN()
        elif self.frg_detect == 'FD':
            self._frg_detect_mth = FrameDifferencing()
        else:
            raise ValueError('Unknown method for foreground detection')

        if self.mat_morph:
            for d in self.mat_morph:
                if 'fun' not in d.keys():
                    raise ValueError('Parameter mat_morph has no key "fun".')
                if 'kernel' not in d.keys():
                    d['kernel'] = cv.getStructuringElement(
                        cv.MORPH_RECT,
                        (5, 5),
                    )

        if 'perimeter' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "perimeter".')
        if not self.post_filter.get('perimeter'):
            pass  # TODO: display window
        if 'anchor_point' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "anchor_point".')
        if 'perspective' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "perspective".')
        if not self.post_filter.get('perspective'):
            pass  # TODO: display window

        if 'presence_max' not in self.int_detect.keys():
            raise ValueError('Parameter int_detect has no key "presence_max".')

        self.tracked_blobs_ = None

        return self

    def detect(self, X):
        """Predict if there is an intrusion in current frame.

        Parameters
        ----------
        X : ndarray of int, shape (n_height, n_width) or \
                (n_height, n_width, n_channel)
            Input frame.

        Returns
        -------
        y : int
            Prediction of intrusion: 1 if intrusion detected, 0 otherwise.
        """
        if self.convert:
            X = cv.cvtColor(X, self.convert)

        # denoising by spatial blurring
        if self.blur:
            X = self.blur.get('fun')(
                X,
                self.blur.get('ksize'),
                self.blur.get('borderType'),
            )

        # foreground detection
        mask = self._frg_detect_mth.apply(X)

        # denoising by mathematical morphology
        if self.mat_morph:
            for d in self.mat_morph:
                mask = d.get('fun')(mask, d.get('kernel'))
        self.foreground_mask_ = mask

        # blob creation from foreground mask
        self._create_blob()

        # blob tracking
        self._track_blob()

        # post-filtering: perimeter, perspective
        self._post_filter()

        # intrusion detection
        y = self._detect_blob()
        return y

    def _create_blob(self, area_min=100, dist_min=10):  #TODO: in parameters ?
        """Create blobs from foreground mask using contour retrieval."""
        # create blobs using contour retrieval
        _, contours, _ = cv.findContours(
            self.foreground_mask_,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_NONE,
        )

        # filter blobs by area
        areas = [cv.contourArea(contour) for contour in contours]
        self.blobs_ = [
            contour for contour, area in zip(contours, areas)
            if area >= area_min
        ]

    def _track_blob(self, abscence_max=3):
        """Track blobs using only distance to centers."""
        n_blobs = len(self.blobs_)
        if self.tracked_blobs_ is None:
            self.tracked_blobs_ = (
                self.blobs_.copy(),
                [1] * n_blobs,
                [0] * n_blobs
            )

        else:
            n_tracked_blobs = len(self.tracked_blobs_[0])
            if n_blobs > 0 and n_tracked_blobs > 0:
                blobs_cent = self._get_center(self.blobs_)
                tracked_blobs_cent = self._get_center(self.tracked_blobs_[0])
                dist = np.atleast_2d(
                    cdist(blobs_cent, tracked_blobs_cent, 'euclidean')
                )
                for i in range(n_blobs):
                    j_min = np.argmin(dist[i])
                    if dist[i, j_min] < 100:  #TODO: use features to pair blobs
                        dist[i, j_min] = -1
                        self.tracked_blobs_[0][j_min] = self.blobs_[i].copy()
                        self.tracked_blobs_[1][j_min] += 1
                        self.tracked_blobs_[2][j_min] = 0
                    else:
                        self.tracked_blobs_[0].append(self.blobs_[i].copy())
                        self.tracked_blobs_[1].append(1)
                        self.tracked_blobs_[2].append(0)
                for j in range(n_tracked_blobs - 1, 0, -1):
                    if np.all(dist[:, j] >= 0):
                        self.tracked_blobs_[1][j] = 0
                        self.tracked_blobs_[2][j] += 1
                        if self.tracked_blobs_[2][j] > abscence_max:
                            del self.tracked_blobs_[0][j]
                            del self.tracked_blobs_[1][j]
                            del self.tracked_blobs_[2][j]

    def _get_center(self, contours):
        """Compute centers of blobs."""
        centers = []
        for contour in contours:
            moments = cv.moments(contour)
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            centers.append([x, y])
        return np.asarray(centers)

    def _detect_blob(self):
        """Detect intrusion blob by blob."""
        if self.tracked_blobs_ is None:
            return 0
        n_tracked_blobs = len(self.tracked_blobs_[0])
        if n_tracked_blobs == 0:
            return 0
        for i in range(n_tracked_blobs):
            if self.tracked_blobs_[1][i] > self.int_detect.get('presence_max'):
                return 1
        else:
            return 0

    def _post_filter(self):  #TODO
        """Post-filter non-intrusions with perimeter and perspective."""
        pass

    def display(self, frame, presence_max=3):
        """On screen display."""
        for i in range(len(self.tracked_blobs_[0])):
            if self.tracked_blobs_[1][i] > self.int_detect.get('presence_max'):
                cv.drawContours(frame, self.tracked_blobs_[0], i, (0, 0, 255))
            else:
                cv.drawContours(frame, self.tracked_blobs_[0], i, (255, 0, 0))
        cv.imshow('Frame', frame)


class FrameDifferencing():
    """Foreground detection by frame differencing.

    Parameters
    ----------
    threshold : float, default=10
        Threshold on absolute difference to detect foreground.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Foreground_detection#Using_frame_differencing
    """  # noqa
    def __init__(self, threshold=5):
        self.threshold = threshold
        self._X = None

    def apply(self, X):
        """Estimate foreground.

        Parameters
        ----------
        X : ndarray
            Input frame.

        Returns
        -------
        X_new : ndarray
            Foreground frame.
        """
        if self._X is None:
            X_new = np.zeros_like(X)

        else:
            diff = cv.absdiff(X, self._X)
            _, X_new = cv.threshold(
                diff,
                self.threshold,
                255,
                cv.THRESH_BINARY,
            )

        if X_new.ndim > 2:
            X_new = np.mean(X_new, axis=-1, dtype=np.uint8, keepdims=False)
        self._X = X

        return X_new
