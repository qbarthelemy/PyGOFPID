"""Methods."""

import cv2 as cv
import numpy as np

from .helpers import (
    read_first_frame,
    plot_lines,
    plot_rectangles,
    get_bottom,
    get_center,
    find_point,
    find_line,
    find_contours,
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

    blur : OpenCV bur filter | None, default={ \
            'fun': cv.GaussianBlur, \
            'ksize': (3, 3), \
            'borderType': cv.BORDER_DEFAULT}
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
            {'fun': cv.erode, \
             'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5))}, \
            {'fun': cv.dilate, \
             'kernel': cv.getStructuringElement(cv.MORPH_RECT, (5, 5))}]
        List of dictionaries containing OpenCV operators for mathematical
        morphology, and their parameters: 'cv.erode' for erosion [Ersn]_, and
        'cv.dilate' for dilation [Dltn]_.
        If None, no processing.

    tracking : dict, default={'factor': 1.5, 'absence_max': 10}
        Dictionary containing parameters for tracking:

        - factor: multiplicative factor of the height of the perspective,
          defining the major radius of the ellipse of the tracking space,
          defining the maximum displacement between two successive frames.
        - absence_max: number of frames of absence before destroying object.

    post_filter : dict, default={ \
            'perimeter': None, \
            'anchor_point': 'bottom', \
            'perspective': None, \
            'perspective_coeff': 0.5, \
            'presence_min': 3, \
            'distance_min': 0.25}
        Dictionary containing parameters for post-filter detections:

        - perimeter: None, or list of points in normalized coordinates.
          If None, perimeter is the whole frame.
        - anchor: point of object, 'center' or 'bottom', to determine if it is
          in the perimeter.
        - perspective: None, or list of four points in normalized coordinates,
          defining the minimal areas of objects to detect. Two boxes represent
          a person in the image (one box in the foreground and a second in the
          background).
          If None, perspective is initialized on a default configuration.
        - perspective_coeff: multiplicative coefficient of tolerance applied on
          the minimal area predicted by perspective.
        - presence_min: number of frames where the object must be present and
          tracked before raising intrusion alarm, to filter transient blobs.
        - distance_min: distance, given in percentage of blob size, the object
          must travel before raising intrusion alarm, to filter motionless
          blobs.

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

    alarm_def : dict, default={'keep_alarm': True, 'duration_min': 10}
        Dictionary containing parameters defining alarm behavior:

        - keep_alarm: boolean defining if a tracked object in alarm once will
          remain always in alarm.
        - duration_min: when an alarm is raised, number of frames during which
          the alarm is maintained whatever happens (to avoid multiple alarms).

    verbose : bool, default=False
        Verbose mode.

    Attributes
    ----------
    foreground_mask_ : ndarray of int, shape (n_height, n_width, n_color)
        Foreground mask.

    blobs_ : list of dict
        Instantaneous blobs created from foreground mask.
        Each blob is a dictionary containing following keys:

        - contour: contour of blob;
        - anchor: anchor point of blob;
        - bottom: bottom point of blob;
        - center: center point of blob.

    tracked_blobs_ : list of dict
        Tracked blobs are created from instantaneous blobs.
        Each tracked blob is a dictionary containing following keys:

        - same keys as an instantaneous blob;
        - center_first: center point of tracked blob when created;
        - presence: presence count;
        - absence: absence count;
        - filter: types of post-filtering (perimeter, perspective, presence,
          distance);
        - already_in_alarm: boolean defining if a tracked object has already
          been in alarm at least once.

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
        tracking={
            'factor': 1.5,
            'absence_max': 10,
        },
        post_filter={
            'perimeter': None,
            'anchor': 'bottom',
            'perspective': None,
            'perspective_coeff': 0.5,
            'presence_min': 3,
            'distance_min': 0.25,
        },
        alarm_def={
            'keep_alarm': True,
            'duration_min': 10,
        },
        verbose=False,
    ):
        self.convert = convert
        self.blur = blur
        self.frg_detect = frg_detect
        self.mat_morph = mat_morph
        self.tracking = tracking
        self.post_filter = post_filter
        self.alarm_def = alarm_def
        self.verbose = verbose

    def initialize(self):
        """Initialize, checking parameters and setting pipeline.

        Returns
        -------
        self : object
            Initialized instance.
        """

        if self.blur:
            if 'fun' not in self.blur.keys():
                raise KeyError('Parameter blur has no key "fun".')
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
                    raise KeyError('Parameter mat_morph has no key "fun".')
                if 'kernel' not in d.keys():
                    d['kernel'] = cv.getStructuringElement(
                        cv.MORPH_RECT,
                        (5, 5),
                    )

        if 'factor' not in self.tracking.keys():
            raise KeyError('Parameter tracking has no key "factor".')
        if 'absence_max' not in self.tracking.keys():
            raise KeyError('Parameter tracking has no key "absence_max".')

        if 'anchor' not in self.post_filter.keys():
            raise KeyError('Parameter post_filter has no key "anchor".')
        if self.post_filter['anchor'] not in ['bottom', 'center']:
            raise ValueError('Parameter anchor must be "bottom" or "center".')
        if 'perspective_coeff' not in self.post_filter.keys():
            raise KeyError(
                'Parameter post_filter has no key "perspective_coeff".')
        if 'presence_min' not in self.post_filter.keys():
            raise KeyError('Parameter post_filter has no key "presence_min".')
        if 'distance_min' not in self.post_filter.keys():
            raise KeyError('Parameter post_filter has no key "distance_min".')
        self._check_perimeter()
        self._check_perspective()

        if 'keep_alarm' not in self.alarm_def.keys():
            raise KeyError('Parameter alarm_def has no key "keep_alarm".')
        if 'duration_min' not in self.alarm_def.keys():
            raise KeyError('Parameter alarm_def has no key "duration_min".')

        self.tracked_blobs_ = []
        self.input_shape_ = None
        self._alarm_system = {
            'already_in_alarm': False,
            'duration': 0,
        }

        return self

    def _check_perimeter(self):
        """Check parameter perimeter."""

        if 'perimeter' not in self.post_filter.keys():
            raise KeyError('Parameter post_filter has no key "perimeter".')

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
            img.shape,
        ).tolist()

        def move_line(event, x, y, flags, params):
            global i_point, i_line
            if event == cv.EVENT_LBUTTONDOWN:
                i_point = find_point((x, y), points, thickness)
            elif event == cv.EVENT_LBUTTONUP and i_point >= 0:
                points[i_point] = [x, y]
            elif event == cv.EVENT_RBUTTONDOWN:
                i_point = find_point((x, y), points, thickness)
                i_line = find_line((x, y), points, thickness)
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
            raise KeyError('Parameter post_filter has no key "perspective".')

        if self.post_filter['perspective'] is None:
            self.post_filter['perspective'] = np.array(
                [[0.1, 0.4], [0.2, 0.8], [0.8, 0.2], [0.85, 0.3]]
            )
        elif isinstance(self.post_filter['perspective'], np.ndarray):
            if self.post_filter['perspective'].shape != (4, 2):
                raise ValueError(
                    'Parameter perspective has not the good shape.')
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
            dtype=np.int32,
        )

        def move_rectangle(event, x, y, flags, params):
            global i_point
            if event == cv.EVENT_LBUTTONDOWN:
                i_point = find_point((x, y), points, thickness)
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
            if self.verbose:
                print("No configuration video or frame defined.")
            img = 100 * np.ones((240, 320, 3), dtype=np.uint8)

        if img is None:
            raise ValueError('Configuration image is None.')

        clone = img.copy()

        thickness = unnormalize_coords(
            np.array([[0.02, 0.02]]),
            img.shape,
            dtype=np.int32,
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
            self._calib_first_frame()
        else:
            if X.shape != self.input_shape_:
                raise ValueError('Input shape has changed.')

        # input frame color conversion
        if self.convert:
            X = cv.cvtColor(X, self.convert)

        # input frame denoising by spatial blurring
        if self.blur:
            X = self.blur.get('fun')(
                X,
                self.blur.get('ksize'),
                self.blur.get('borderType'),
            )

        # foreground detection
        mask = self._frg_detect_mth.apply(X)

        # foreground mask denoising by mathematical morphology
        if self.mat_morph:
            for d in self.mat_morph:
                mask = d.get('fun')(mask, d.get('kernel'))
        self.foreground_mask_ = mask

        # blob creation from foreground mask
        self._find_blob()

        # blob tracking and some post-filtering
        self._track_blob()

        # post-filtering: perimeter, perspective
        self._post_filter()

        # intrusion detection
        y = self._detect_blob()

        # alarm enforcement according to alarm definition
        y = self._maintain_alarm(y)

        return y

    def _calib_first_frame(self):
        """Finish configurations using first frame dimensions."""

        # calibrate perimeter
        self.post_filter['perimeter'] = unnormalize_coords(
            self.post_filter['perimeter'],
            self.input_shape_,
            dtype=np.int32,
        )

        # calibrate perspective
        self.post_filter['perspective'] = unnormalize_coords(
            self.post_filter['perspective'],
            self.input_shape_,
            dtype=np.float64,
        )
        self._calib_perspective()

    def _calib_perspective(self):
        """Calibrate perspective:
        area, height and width as a function of bottom point.
        """

        points = self.post_filter['perspective']
        bottoms = [
            np.max([points[1][1], points[0][1]]),
            np.max([points[3][1], points[2][1]]),
        ]

        # calibrate area as a function of bottom point
        areas = [
            abs((points[1][1] - points[0][1]) * (points[1][0] - points[0][0])),
            abs((points[3][1] - points[2][1]) * (points[3][0] - points[2][0])),
        ]
        slr = SimpleLinearRegression().fit(bottoms, areas)
        y = np.arange(0, self.input_shape_[0], 1)
        self._perspective = {'area': slr.predict_clip(y, 0)}

        # calibrate height as a function of bottom point
        heights = [
            abs(points[1][1] - points[0][1]),
            abs(points[3][1] - points[2][1]),
        ]
        slr.fit(bottoms, heights)
        self._perspective['height'] = slr.predict_clip(y, 0)

        # calibrate width as a function of bottom point
        widths = [
            abs(points[1][0] - points[0][0]),
            abs(points[3][0] - points[2][0]),
        ]
        slr.fit(bottoms, widths)
        self._perspective['width'] = slr.predict_clip(y, 0)

    def _predict_perspective(self, key, y):
        """Predict perspective."""

        y = np.clip(y, a_min=0, a_max=self.input_shape_[0] - 1)
        return self._perspective.get(key)[y]

    def _find_blob(self, area_min=100):  #TODO: in parameters ?
        """Find blobs from foreground mask using contour retrieval."""

        # find blobs using contour retrieval
        contours = find_contours(self.foreground_mask_)

        # filter contours with minimal area
        contours = [
            contour for contour in contours
            if cv.contourArea(contour) >= area_min
        ]

        # create instantaneous blobs
        self.blobs_ = [
            self._create_blob(contour) for contour in contours
        ]

    def _create_blob(self, contour):
        """Create blob from its contour."""

        bottom = get_bottom(contour, dtype=np.int16)
        center = get_center(contour, dtype=np.int16)
        if self.post_filter['anchor'] == 'bottom':
            anchor = bottom
        else:
            anchor = center

        blob = {
            'contour': contour,
            'anchor': anchor,
            'bottom': bottom,
            'center': center,
        }
        return blob

    def _track_blob(self):
        """Track blobs.

        Current tracking uses only distance between centers.
        #TODO: compute similarities using features extracted on contours.
        """

        n_tracked_blobs = len(self.tracked_blobs_)
        n_blobs = len(self.blobs_)

        if n_tracked_blobs == 0:
            for i in range(n_blobs):
                self.tracked_blobs_.append(self._create_tracked_blob(i))
            return
        if n_blobs == 0:
            for i in range(n_tracked_blobs - 1, -1, -1):
                self._update_unpaired_tracked_blob(i)
            return

        # compute distances between centers of blobs
        dist = np.zeros((n_blobs, n_tracked_blobs))
        for i in range(n_blobs):
            for j in range(n_tracked_blobs):
                dist[i, j] = self._compute_tracking_distance(j, i)

        # find tracked blob corresponding to each instantaneous blob
        for i in range(n_blobs):
            #TODO: use similarities to find the optimal blob
            j_min = np.argmin(dist[i])
            if dist[i, j_min] <= 1:  # pair found
                dist[i, j_min] = -1  # to mark that pair has been found
                self._update_paired_tracked_blob(j_min, i)
            else:
                self.tracked_blobs_.append(self._create_tracked_blob(i))

        # process unpaired tracked blobs
        for j in range(n_tracked_blobs - 1, -1, -1):
            if np.all(dist[:, j] >= 0):  # no pair found
                self._update_unpaired_tracked_blob(j)

    def _create_tracked_blob(self, i_blob):
        """Create a tracked blob."""

        blob = self.blobs_[i_blob]

        tracked_blob = {
            'contour': blob['contour'].copy(),
            'anchor': blob['anchor'],
            'bottom': blob['bottom'],
            'center': blob['center'],
            'center_first': blob['center'],
            'presence': 1,
            'absence': 0,
            'filtered': set(['presence', 'distance']),
            'already_in_alarm': False,
        }
        return tracked_blob

    def _compute_tracking_distance(self, i_tracked_blob, i_blob):
        """Compute tracking distance.

        Compute the tracking distance between a tracked blob (tb) and an
        instantaneous blob (ib).
        The instantaneous blob belongs to the tracking space if its distance to
        the ellipse is inferior to 1:
        dist = (x_ib - x_tb)^2 / r_maj^2 + (y_ib - y_tb)^2 / r_min^2 <= 1
        """

        tracked_blob_bottom = self.tracked_blobs_[i_tracked_blob]['bottom']
        radius_maj, radius_min = self._compute_tracking_space(tracked_blob_bottom)

        tracked_blob_center = self.tracked_blobs_[i_tracked_blob]['center']
        blob_center = self.blobs_[i_blob]['center']

        if radius_maj <= 0 or radius_min <= 0:
            return 0

        dist = (blob_center[0] - tracked_blob_center[0])**2 / radius_maj**2 \
            + (blob_center[1] - tracked_blob_center[1])**2 / radius_min**2
        return dist

    def _compute_tracking_space(self, bottom):
        """Compute major and minor radii of the ellipse of tracking space."""

        height_min = self._predict_perspective('height', bottom[1])
        radius_major = np.int32(height_min * self.tracking['factor'])
        radius_minor = np.int32(radius_major / 2)
        return radius_major, radius_minor

    def _update_paired_tracked_blob(self, i_tracked_blob, i_blob):
        """Update a tracked blob paired with an instantaneous blob."""

        tracked_blob = self.tracked_blobs_[i_tracked_blob]
        blob = self.blobs_[i_blob]

        tracked_blob['contour'] = blob['contour'].copy()
        tracked_blob['anchor'] = blob['anchor']
        tracked_blob['bottom'] = blob['bottom']
        tracked_blob['center'] = blob['center']
        tracked_blob['absence'] = 0
        tracked_blob['presence'] += 1

        if tracked_blob['presence'] >= self.post_filter['presence_min']:
            tracked_blob['filtered'].discard('presence')
        if self._compute_total_distance(i_tracked_blob) >= self.post_filter['distance_min']:
            tracked_blob['filtered'].discard('distance')
        #else:
        #    tracked_blob['filtered'].add('distance') # Q: seems dangerous if circular mouvement?

    def _compute_total_distance(self, i_tracked_blob):
        """Compute total distance between centers of first and last positions."""

        # distance between first and last centers
        dist = cv.norm(
            self.tracked_blobs_[i_tracked_blob]['center']
            - self.tracked_blobs_[i_tracked_blob]['center_first']
        )

        # perspective size at the position of tracked blob
        area_min = self._predict_perspective(
            'area',
            self.tracked_blobs_[i_tracked_blob]['bottom'][1],
        )
        if area_min <= 0:
            return 0

        # relative distance: distance normalized by perspective size
        # => filtering distance is a function of depth
        dist_rel = dist / np.sqrt(area_min)
        return dist_rel

    def _update_unpaired_tracked_blob(self, i_tracked_blob):
        """Update an unpaired tracked blob."""

        self.tracked_blobs_[i_tracked_blob]['presence'] = 0
        self.tracked_blobs_[i_tracked_blob]['filtered'].add('presence')
        self.tracked_blobs_[i_tracked_blob]['absence'] += 1

        if self.tracked_blobs_[i_tracked_blob]['absence'] >= self.tracking['absence_max']:
            self.tracked_blobs_.pop(i_tracked_blob)

    def _post_filter(self):
        """Post-filter pre-alarms with perimeter and perspective."""

        for tracked_blob in self.tracked_blobs_:
            # filtering by perimeter
            if cv.pointPolygonTest(
                self.post_filter['perimeter'],
                tracked_blob['anchor'],
                False,
            ) < 0:
                # object not in perimeter => filtered
                tracked_blob['filtered'].add('perimeter')
            else:
                # object in perimeter
                tracked_blob['filtered'].discard('perimeter')

            # filtering by perspective
            _, _, width, height = cv.boundingRect(tracked_blob['contour'])
            area = width * height
            area_persp = self._predict_perspective(
                'area',
                tracked_blob['bottom'][1],
            )
            if area <= area_persp * self.post_filter['perspective_coeff']:
                # object is smaller than minimal perspective => filtered
                tracked_blob['filtered'].add('perspective')
            else:
                # object of interest
                tracked_blob['filtered'].discard('perspective')

    def _detect_blob(self):
        """Detect intrusion analyzing tracked blobs."""

        blob_in_alarm = False

        for tracked_blob in self.tracked_blobs_:
            if tracked_blob['filtered'] == set():
                # not filtered => intrusion detected
                blob_in_alarm = True
                if self.alarm_def['keep_alarm']:
                    tracked_blob['already_in_alarm'] = True
            elif tracked_blob['already_in_alarm']:
                blob_in_alarm = True
            else:
                pass

        return blob_in_alarm

    def _maintain_alarm(self, sys_in_alarm):
        """Maintain alarm according to alarm definition."""

        self._alarm_system['duration'] -= 1

        if sys_in_alarm and not self._alarm_system['already_in_alarm']: # and self._alarm_system.duration <= 0:
            # beginning of intrusion
            self._alarm_system['duration'] = self.alarm_def['duration_min']

        if not sys_in_alarm and self._alarm_system['duration'] >= 0:
            sys_in_alarm = True

        if self.verbose and sys_in_alarm and not self._alarm_system['already_in_alarm']:
            print("Intrusion detected")

        self._alarm_system['already_in_alarm'] = sys_in_alarm

        return sys_in_alarm

    def display(self, X, display_tracking=False, display_perspective=False):
        """On screen display.

        Parameters
        ----------
        X : ndarray of int, shape (n_height, n_width) or \
                (n_height, n_width, n_channel)
            Input frame.
        display_tracking : bool, default=False
            Flag to display the tracking space of tracked blobs.
        display_perspective : bool, default=False
            Flag to display the perspective of tracked blobs.
        """

        cv.drawContours(X, [self.post_filter['perimeter']], 0, (25, 200, 200))

        for tracked_blob in self.tracked_blobs_:

            if tracked_blob['filtered'] == set() or tracked_blob['already_in_alarm']:
                color = (0, 0, 255)
            elif 'perimeter' in tracked_blob['filtered']:
                color = (255, 0, 0)
            elif 'perspective' in tracked_blob['filtered']:
                color = (25, 200, 200)
            elif 'presence' in tracked_blob['filtered'] \
                    or 'distance' in tracked_blob['filtered']:
                color = (0, 255, 0)
            else:
                raise ValueError(f'Unknown filtering type {tracked_blob["filtered"]}')

            cv.drawContours(X, [tracked_blob['contour']], 0, color)
            cv.circle(X, tracked_blob['anchor'], 4, color, -1)

            if display_tracking:
                self._display_tracking(
                    X, tracked_blob['bottom'], tracked_blob['center'], color
                )
            if display_perspective:
                self._display_perspective(X, tracked_blob['bottom'], color)

    def _display_tracking(self, X, bottom, center, color):
        """Display tracking space."""
        radius_maj, radius_min = self._compute_tracking_space(bottom)
        cv.ellipse(X, center, [radius_maj, radius_min], 0, 0, 360, color, 1)

    def _display_perspective(self, X, bottom, color):
        """Display perspective."""
        height = np.int32(self._predict_perspective('height', bottom[1]))
        width = np.int32(self._predict_perspective('width', bottom[1]))
        rect1 = [bottom[0] - width//2, bottom[1]]
        rect2 = [bottom[0] + width//2, bottom[1] - height]
        cv.rectangle(X, rect1, rect2, color, 1)


###############################################################################


class FrameDifferencing():
    """Foreground detection by frame differencing.

    F = abs(X_t - X_{t-1}) > threshold

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
        F : ndarray
            Foreground frame.
        """

        if self._X is None:
            F = np.zeros_like(X)
        else:
            diff = cv.absdiff(X, self._X)
            _, F = cv.threshold(
                diff,
                self.threshold,
                255,
                cv.THRESH_BINARY,
            )

        self._X = X
        if F.ndim > 2:
            F = np.mean(F, axis=-1, dtype=np.uint8, keepdims=False)

        return F
