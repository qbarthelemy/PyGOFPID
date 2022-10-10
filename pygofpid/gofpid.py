
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
        Dictionary containing parameters to filter non-intrusions:

        - perimeter: list of points. If None, a window allows to draw it.
        - anchor: 'center' or 'bottom', to determine if an object is in the
          perimeter.
        - perspective: list of four points, defining the minimum sizes of
          objects to detect. Two boxes represent a person in the image (one box
          in the foreground and a second in the background).
          If None, a window allows to draw these rectangles according to the
          size of a person placed in these places in the image.

    int_detect : dict, default={'presence_max': 3}
        Dictionary containing parameters to detect intrusion:

        - presence_max: number of frames where objet is present and tracked
          before raising intrusion alarm.

    Attributes
    ----------
    foreground_mask_ : ndarray of int, shape (n_height, n_width, n_color)
        Foreground mask.

    blobs_ : list of n_blobs lists of n_points lists of two int
        Instantaneous blobs created from foreground mask.

    tracked_blobs_ : list of n_blobs dict
        Tracked blobs created from instantaneous blobs:

        - contour: contour of blob;
        - presence: presence count;
        - absence: absence count;
        - filter: typf of filtering.

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
            'anchor': 'center',
            'perspective': None,
        },
        int_detect={'presence_max': 3},
        verbose=False
    ):
        self.convert = convert
        self.blur = blur
        self.frg_detect = frg_detect
        self.mat_morph = mat_morph
        self.post_filter = post_filter
        self.int_detect = int_detect
        self.verbose = verbose
        self.input_shape = None

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
        if not isinstance(self.post_filter['perimeter'], np.ndarray):
            self.post_filter['perimeter'] = self._display_perimeter()
        if 'anchor' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "anchor".')
        if self.post_filter.get('anchor') == 'center':
            self._get_anchors = get_centers
        elif self.post_filter.get('anchor') == 'bottom':
            self._get_anchors = get_bottoms
        else:
            raise ValueError('Parameter anchor must be "center" or "bottom".')
        if 'perspective' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "perspective".')
        if not isinstance(self.post_filter['perspective'], np.ndarray):
            self.post_filter['perspective'] = self._display_perspective()
        if 'presence_max' not in self.int_detect.keys():
            raise ValueError('Parameter int_detect has no key "presence_max".')

        self.tracked_blobs_ = None

        return self

# buttons for reset and ok
# allow to use an image from video

    def _display_perimeter(self):

        img = np.random.randint(0, high=255, size=(200, 300, 3), dtype=np.uint8)
        clone = img.copy()
        perimeter = []

        def draw_line(event, x, y, flags, param):
            if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONUP]:
                perimeter.append([x, y])
                if len(perimeter) > 1:
                    cv.line(img, perimeter[-2], perimeter[-1], (0, 0, 255), 2)
                    cv.imshow("Config perimeter", img)

        cv.namedWindow("Config perimeter")
        cv.setMouseCallback("Config perimeter", draw_line)

        # keep looping until the 'c' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow("Config perimeter", img)
            key = cv.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset image
            if key == ord("r"):
                img = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
        cv.destroyWindow("Config perimeter")

        perimeter = np.asarray(perimeter, dtype=np.float64)
        perimeter[:, 0] /= img.shape[1]
        perimeter[:, 1] /= img.shape[0]
        if self.verbose:
            print("Config perimeter:\n", perimeter)
        return perimeter

# TODO: display rectangles, and allow to modify them
    def _display_perspective(self):

        img = np.random.randint(0, high=255, size=(200, 300, 3), dtype=np.uint8)
        clone = img.copy()
        rects = []

        def draw_rectangle(event, x, y, flags, param):
            global rect
            if event == cv.EVENT_LBUTTONDOWN:
                rect = [[x, y]]
            elif event == cv.EVENT_LBUTTONUP:
                rect.append([x, y])
                cv.rectangle(img, rect[0], rect[1], (0, 0, 255), 2)
                cv.imshow("Config perspective", img)
                rects.append(rect)

        cv.namedWindow("Config perspective")
        cv.setMouseCallback("Config perspective", draw_rectangle)

        # keep looping until the 'c' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow("Config perspective", img)
            key = cv.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset image
            if key == ord("r"):
                img = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
        cv.destroyWindow("Config perspective")

        # TODO: normalized coordinates

        if self.verbose:
            print("Config perspective:\n", rects)
        return rects

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
        # input shape checking
        if not self.input_shape:
            self.input_shape = X.shape
        else:
            if X.shape != self.input_shape:
                raise ValueError('Input shape has changed.')

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

    def _track_blob(self, absence_max=3):
        """Track blobs using only distance to centers."""
        n_blobs = len(self.blobs_)
        if self.tracked_blobs_ is None:
            self.tracked_blobs_ = []
            for i in range(n_blobs):
                self.tracked_blobs_.append({
                    'contour': self.blobs_[i].copy(),
                    'presence': 1,
                    'absence': 0,
                    'filter': 'presence',
                })

        else:
            n_tracked_blobs = len(self.tracked_blobs_)
            if n_blobs > 0 and n_tracked_blobs > 0:
                # parwise distances
                blobs_cent = get_centers(self.blobs_)
                tracked_blobs_cent = get_centers(
                    [blob['contour'] for blob in self.tracked_blobs_]
                )
                dist = np.atleast_2d(
                    cdist(blobs_cent, tracked_blobs_cent, 'euclidean')
                )

                for i in range(n_blobs):
                    j_min = np.argmin(dist[i])
                    #TODO: use features to pair blobs
                    if dist[i, j_min] < 100:  # pair found
                        dist[i, j_min] = -1
                        self.tracked_blobs_[j_min]['contour'] = self.blobs_[i].copy()
                        self.tracked_blobs_[j_min]['presence'] += 1
                        self.tracked_blobs_[j_min]['absence'] = 0
                    else:
                        self.tracked_blobs_.append({
                            'contour': self.blobs_[i].copy(),
                            'presence': 1,
                            'absence': 0,
                            'filter': 'presence',
                        })

                for i in range(n_tracked_blobs - 1, 0, -1):
                    if np.all(dist[:, i] >= 0): # no pair found
                        self.tracked_blobs_[i]['presence'] = 0
                        self.tracked_blobs_[i]['absence'] += 1
                        if self.tracked_blobs_[i]['absence'] > absence_max:
                            self.tracked_blobs_.pop(i)

    def _post_filter(self):
        """Post-filter non-intrusions with perimeter and perspective."""
        # perimeter
        perimeter = self.post_filter['perimeter']
        if np.all((0.0 <= perimeter) & (perimeter <= 1.0)):
            perimeter[:, 0] *= self.input_shape[1]
            perimeter[:, 1] *= self.input_shape[0]
            self.post_filter['perimeter'] = perimeter.astype(np.int32)

        anchors = self._get_anchors(
            [blob['contour'] for blob in self.tracked_blobs_]
        )
        for i in range(len(anchors)):
            if cv.pointPolygonTest(
                self.post_filter['perimeter'],
                (anchors[i][0], anchors[i][1]),
                False,
            ) < 0:  # object not in perimeter
                self.tracked_blobs_[i]['filter'] = 'perimeter'

        # perspective  #TODO

    def _detect_blob(self):
        """Detect intrusion blob by blob."""
        if self.tracked_blobs_ is None:
            return 0

        y = 0
        for i, blob in enumerate(self.tracked_blobs_):
            if blob['presence'] > self.int_detect.get('presence_max'):
                self.tracked_blobs_[i]['filter'] = None
                y = 1

        return y

    def display(self, X):
        """On screen display.

        Parameters
        ----------
        X : ndarray of int, shape (n_height, n_width) or \
                (n_height, n_width, n_channel)
            Input frame.
        """
        cv.drawContours(X, [self.post_filter['perimeter']], 0, (25, 200, 200))

        for blob in self.tracked_blobs_:
            if blob['filter'] == 'presence':
                cv.drawContours(X, [blob['contour']], 0, (0, 255, 0))
            elif blob['filter'] == 'perimeter':
                cv.drawContours(X, [blob['contour']], 0, (255, 0, 0))
            elif blob['filter'] == None:
                cv.drawContours(X, [blob['contour']], 0, (0, 0, 255))
            else:
                raise ValueError('Unknown filtering type')

        cv.imshow('Frame', X)


###############################################################################

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


###############################################################################

def get_centers(contours):
    """Compute centers of contours."""
    centers = []
    for contour in contours:
        moments = cv.moments(contour)
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        centers.append([x, y])
    return np.array(centers, dtype=np.uint8)


def get_bottoms(contours):
    """Compute middle-bottom points of contours."""
    bottoms = []
    for contour in contours:
        moments = cv.moments(contour)
        x = int(moments["m10"] / moments["m00"])
        y = max(contour[:, 1])
        bottoms.append([x, y])
    return np.array(bottoms, dtype=np.uint8)
