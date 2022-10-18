"""Methods."""

import numpy as np
from scipy.spatial.distance import cdist
import cv2 as cv

from .helpers import (
    read_first_frame,
    plot_lines,
    plot_rectangles,
    get_centers,
    get_bottoms,
    is_in_squares,
    is_between_points,
    normalize_coords,
    unnormalize_coords,
    SimpleLinearRegression,
)


class GOFPID():
    """GOFPID: good old fashioned perimeter intrusion detection system.

    Leveraging OpenCV, this class allows to build a pipeline using good old
    fashioned (GOF) computer vision methods for perimeter intrusion detection
    (PID):

    1. input frame denoising by spatial blurring;
    2. foreground detection by background subtraction;
    3. foreground mask denoising by mathematical morphology;
    4. foreground blob creation;
    5. blob tracking (WIP);
    6. post-filtering (perimeter, perspective, presence);
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

    post_filter : dict, default={'perimeter': None, 'anchor_point': 'bottom', \
            'perspective': None, 'perspective_coeff': 0.5, 'presence_max': 3}
        Dictionary containing parameters to filter pre-alarms:

        - perimeter: None, or list of points in normalized coordinates.
          If None, perimeter is the whole frame.
        - anchor: point of object, 'center' or 'bottom', to determine if it is
          in the perimeter.
        - perspective: list of four points in normalized coordinates, defining
          the minimum sizes of objects to detect. Two boxes represent a person
          in the image (one box in the foreground and a second in the
          background).
          If None, perspective is.
        - perspective_coeff: multiplicative coefficient of tolerance on size of
          perspective.
        - presence_max: number of frames where objet is present and tracked
          before raising intrusion alarm.

        - display_config, optional: choose if display configuration.
        - config_video_filename, optional: filename of video to display its
          first frame to configure perimeter and perspective.
        - config_frame_filename, optional: filename of frame to display to
          configure perimeter and perspective.
        - config_frame, optional: frame to display to configure perimeter and
          perspective.

        On configuration windows, press 'r' key to reset, 'c' to close.
        For perimeter, right-click on a line to add a new point, and on an
        existing point to remove it.
        For perspective, a window allows to draw these rectangles according to
        the size of a person placed in these places in the image.

    Attributes
    ----------
    foreground_mask_ : ndarray of int, shape (n_height, n_width, n_color)
        Foreground mask.

    blobs_ : list of OpenCV contours
        Instantaneous blobs created from foreground mask.

    tracked_blobs_ : list of n_blobs dict
        Tracked blobs created from instantaneous blobs:

        - contour: contour of blob;
        - presence: presence count;
        - absence: absence count;
        - filter: types of filtering.

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
            'anchor': 'bottom',
            'perspective': None,
            'perspective_coeff': 0.5,
            'presence_max': 3,
        },
        verbose=False
    ):
        self.convert = convert
        self.blur = blur
        self.frg_detect = frg_detect
        self.mat_morph = mat_morph
        self.post_filter = post_filter
        self.verbose = verbose

    def init(self):
        """Initialize, checking parameters and setting pipeline.

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

        if 'anchor' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "anchor".')
        if self.post_filter.get('anchor') == 'center':
            self._get_anchors = get_centers
        elif self.post_filter.get('anchor') == 'bottom':
            self._get_anchors = get_bottoms
        else:
            raise ValueError('Parameter anchor must be "center" or "bottom".')
        if 'perspective_coeff' not in self.post_filter.keys():
            raise ValueError(
                'Parameter post_filter has no key "perspective_coeff".')
        if 'presence_max' not in self.post_filter.keys():
            raise ValueError(
                'Parameter post_filter has no key "presence_max".')
        self._check_perimeter()
        self._check_perspective()
        self._calib_perspective()

        self.tracked_blobs_ = None
        self.input_shape_ = None

        return self

    def _check_perimeter(self):
        """Check parameter perimeter."""
        if 'perimeter' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "perimeter".')

        if self.post_filter['perimeter'] is None:
            self.post_filter['perimeter'] = np.array(
                [[0, 0], [0, 1], [1, 1], [1, 0]]
            )
        elif isinstance(self.post_filter['perimeter'], np.ndarray):
            if self.post_filter['perimeter'].ndim != 2:
                raise ValueError('Parameter perimeter has not the good shape.')
            if self.post_filter['perimeter'].shape[0] < 3:
                raise ValueError('Parameter perimeter has not enough points.')
            if self.post_filter['perimeter'].shape[1] != 2:
                raise ValueError('Parameter perimeter has not the good shape.')
        else:
            raise ValueError('Unknown type for parameter "perimeter".')

        if self.post_filter.get('display_config'):
            self.post_filter['perimeter'] = self._config_perimeter()

    def _config_perimeter(self, window_name="Configure perimeter"):
        """Display window to configure perimeter."""
        img, clone, thickness = self._get_config_img()

        points = unnormalize_coords(
            self.post_filter['perimeter'],
            img.shape
        ).tolist()

        def move_line(event, x, y, flags, params):
            global i_point, i_line
            if event == cv.EVENT_LBUTTONDOWN:
                i_point = is_in_squares((x, y), points, thickness)
            elif event == cv.EVENT_LBUTTONUP and i_point >= 0:
                points[i_point] = [x, y]
            elif event == cv.EVENT_RBUTTONDOWN:
                i_point = is_in_squares((x, y), points, thickness)
                i_line = is_between_points((x, y), points, thickness)
            elif event == cv.EVENT_RBUTTONUP:
                if i_point >= 0:
                    points.pop(i_point)
                elif i_line >= 0:
                    points.insert(i_line + 1, [x, y])

        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, move_line)
        while True:
            img = clone.copy()
            plot_lines(img, points, thickness)
            cv.imshow(window_name, img)
            key = cv.waitKey(1) & 0xFF
            if key == ord("r"):  # 'r' key => reset window
                img = clone.copy()
                points = unnormalize_coords(
                    np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
                    img.shape,
                ).tolist()
            elif key == ord("c"):  # 'c' key => close
                cv.destroyWindow(window_name)
                break

        perimeter = normalize_coords(points, img.shape)
        if self.verbose:
            print("Config perimeter:\n", perimeter)
        return perimeter

    def _check_perspective(self):
        """Check parameter perspective."""
        if 'perspective' not in self.post_filter.keys():
            raise ValueError('Parameter post_filter has no key "perspective".')

        if self.post_filter['perspective'] is None:
            self.post_filter['perspective'] = np.array(
                [[0.1, 0.4], [0.2, 0.8], [0.8, 0.2], [0.85, 0.3]]
            )
        elif isinstance(self.post_filter['perspective'], np.ndarray):
            if self.post_filter['perspective'].shape != (4, 2):
                 raise ValueError('Parameter perspective has not the good shape.')
        else:
            raise ValueError('Unknown type for parameter "perspective".')

        if self.post_filter.get('display_config'):
            self.post_filter['perspective'] = self._config_perspective()

    def _config_perspective(self, window_name="Configure perspective"):
        """Display window to configure perspective."""
        img, clone, thickness = self._get_config_img()

        points = unnormalize_coords(
            self.post_filter['perspective'],
            img.shape,
            dtype=np.int32
        )

        def move_rectangle(event, x, y, flags, params):
            global i_point
            if event == cv.EVENT_LBUTTONDOWN:
                i_point = is_in_squares((x, y), points, thickness)
            elif event == cv.EVENT_LBUTTONUP and i_point >= 0:
                points[i_point] = [x, y]

        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, move_rectangle)
        while True:
            img = clone.copy()
            plot_rectangles(img, points, thickness)
            cv.imshow(window_name, img)
            key = cv.waitKey(1) & 0xFF
            if key == ord("c"):  # 'c' key => close
                cv.destroyWindow(window_name)
                break

        perspective = normalize_coords(points, img.shape)
        if self.verbose:
            print("Config perspective:\n", perspective)
        return perspective

    def _get_config_img(self):
        if 'config_video_filename' in self.post_filter.keys():
            img = read_first_frame(self.post_filter['config_video_filename'])
        elif 'config_frame_filename' in self.post_filter.keys():
            img = cv.imread(self.post_filter['config_frame_filename'])
        elif 'config_frame' in self.post_filter.keys():
            img = self.post_filter['config_frame']
        else:
            img = 100 * np.ones((240, 320, 3), dtype=np.uint8)

        if img is None:
            raise ValueError('Configuration image is None.')

        clone = img.copy()

        thickness = unnormalize_coords(
            np.array([[0.02, 0.02]]),
            img.shape,
            dtype=np.int32
        )[0]

        return img, clone, thickness

    def detect(self, X):
        """Detect if there is an intrusion in the current frame.

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
        if not self.input_shape_:
            self.input_shape_ = X.shape
            # finish perimeter configuration using first frame
            self.post_filter['perimeter'] = unnormalize_coords(
                self.post_filter['perimeter'],
                self.input_shape_,
                dtype=np.int32
            )
        else:
            if X.shape != self.input_shape_:
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

    def _calib_perspective(self):
        """Calibrate perspective using linear regression on blob height."""
        rects = self.post_filter['perspective']
        self._perspective = SimpleLinearRegression().fit(
            [max(rects[1][1], rects[0][1]), max(rects[3][1], rects[2][1])],
            [abs(rects[1][1] - rects[0][1]), abs(rects[3][1] - rects[2][1])]
        )

    def _create_blob(self, area_min=100): #TODO: in parameters ?
        """Create blobs from foreground mask using contour retrieval."""
        # create blobs using contour retrieval
        _, contours, _ = cv.findContours(
            self.foreground_mask_,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_NONE,
        )

        # filter blobs by area
        self.blobs_ = [
            contour for contour in contours
            if cv.contourArea(contour) >= area_min
        ]

    #TODO: WIP, because naive tracking
    def _track_blob(self, dist_max=100, absence_max=3): #TODO: in parameters ?
        """Track blobs using only distance to centers."""
        n_blobs = len(self.blobs_)
        if self.tracked_blobs_ is None:
            self.tracked_blobs_ = []
            for i in range(n_blobs):
                self.tracked_blobs_.append({
                    'contour': self.blobs_[i].copy(),
                    'presence': 1,
                    'absence': 0,
                    'filter': set(['presence']),
                })

        else:
            n_tracked_blobs = len(self.tracked_blobs_)
            if n_blobs > 0 and n_tracked_blobs > 0:
                # pairwise distances
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
                    if dist[i, j_min] < dist_max:  # pair found
                        dist[i, j_min] = -1
                        self.tracked_blobs_[j_min]['contour'] = self.blobs_[i].copy()
                        self.tracked_blobs_[j_min]['absence'] = 0
                        self.tracked_blobs_[j_min]['presence'] += 1
                        if self.tracked_blobs_[j_min]['presence'] >= self.post_filter['presence_max']:
                            self.tracked_blobs_[j_min]['filter'].discard('presence')
                    else:
                        self.tracked_blobs_.append({
                            'contour': self.blobs_[i].copy(),
                            'presence': 1,
                            'absence': 0,
                            'filter': set(['presence']),
                        })

                for i in range(n_tracked_blobs - 1, 0, -1):
                    if np.all(dist[:, i] >= 0): # no pair found
                        self.tracked_blobs_[i]['presence'] = 0
                        self.tracked_blobs_[i]['filter'].add('presence')
                        self.tracked_blobs_[i]['absence'] += 1
                        if self.tracked_blobs_[i]['absence'] >= absence_max:
                            self.tracked_blobs_.pop(i)

    def _post_filter(self, perspective_coeff=0.75):
        """Post-filter pre-alarms with perimeter and perspective."""
        anchors = self._get_anchors(
            [blob['contour'] for blob in self.tracked_blobs_],
            dtype=np.int16
        )

        for i in range(len(self.tracked_blobs_)):
            # perimeter
            if cv.pointPolygonTest(
                self.post_filter['perimeter'],
                anchors[i],
                False,
            ) < 0:  # object not in perimeter
                self.tracked_blobs_[i]['filter'].add('perimeter')
            else:  # object in perimeter
                self.tracked_blobs_[i]['filter'].discard('perimeter')

            # perspective
            _, _, _, height = cv.boundingRect(
                self.tracked_blobs_[i]['contour']
            )
            height_min = self._perspective.predict(anchors[i][1])
            if height <= height_min * self.post_filter['perspective_coeff']:  # object too small
                self.tracked_blobs_[i]['filter'].add('perspective')
            else:  # object acceptable
                self.tracked_blobs_[i]['filter'].discard('perspective')

    def _detect_blob(self):
        """Detect intrusion blob by blob."""
        if self.tracked_blobs_ is None:
            return 0

        y = 0
        for i, blob in enumerate(self.tracked_blobs_):
            if blob['filter'] == set():  # no filter => intrusion detected
                y = 1

        if self.verbose and y == 1:
            print("Intrusion detected")
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
            if blob['filter'] == set():
                color = (0, 0, 255)
            elif 'perimeter' in blob['filter']:
                color = (255, 0, 0)
            elif 'perspective' in blob['filter']:
                color = (25, 200, 200)
            elif 'presence' in blob['filter']:
                color = (0, 255, 0)
            else:
                raise ValueError('Unknown filtering type')
            cv.drawContours(X, [blob['contour']], 0, color)
            anchor = self._get_anchors([blob['contour']], dtype=np.int16)[0]
            cv.circle(X, anchor, 4, color, -1)

        cv.imshow('Frame', X)


###############################################################################


class FrameDifferencing():
    """Foreground detection by frame differencing.

    Parameters
    ----------
    threshold : float, default=5
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
