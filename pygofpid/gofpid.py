
import numpy as np
import cv2 as cv


class GOFPID:
    """GOFPID: good old fashioned perimeter intrusion detection system.

    This class allows to build a pipeline using good old fashioned (GOF)
    computer vision methods for perimeter intrusion detection (PID):

    1 - input frame denoising by spatial blurring;
    2 - foreground segmentation by background subtraction;
    3 - tracking;
    4 - motion mask denoising by mathematical morphology.

    Parameters
    ----------
    blur : OpenCV bur filter, default='GaussianBlur'
        Filter for spatial blurring of input frames:

        - 'blur' for a normalized box filter [BoxBlur]_;
        - 'GaussianBlur' for a Gaussian filter [GaussBlur]_;
        - None, to not blur input frame.

    blur_ksize : None | tuple of int, default None
        Size of kernel for spatial blurring.
        If None, it uses a 3x3 square kernel.

    frg_seg : OpenCV BackgroundSubtractor, default='MOG2'
        Method for background subtraction [BkgSub]_:

        - 'MOG2' for mixture of Gaussians [MOG2]_;
        - 'KNN' for K-nearest neigbours [KNN]_.

    mat_morph : list of str, default=['erode', 'dilate']
        List of operators for mathematical morphology, among
        'erode' for erosion [Ersn]_ and 'dilate' for dilation [Dltn]_.

    mat_morph_kernel : None | ndarray of int, shape (n_size_h, n_size_w), \
            default None
        Kernel for mathematical morphology.
        If None, it uses a 5x5 square kernel.

    Attributes
    ----------
    motion_mask_ : ndarray of int, shape (n_height, n_width, n_color)
        Motion mask.

    References
    ----------
    .. [BoxBlur] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37
    .. [GaussBlur] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    .. [BkgSub] https://docs.opencv.org/3.4.0/d7/df6/classcv_1_1BackgroundSubtractor.html
    .. [MOG2] https://docs.opencv.org/3.4.0/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
    .. [KNN] https://docs.opencv.org/3.4.0/db/d88/classcv_1_1BackgroundSubtractorKNN.html
    .. [Ersn] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
    .. [Dltn] https://docs.opencv.org/3.4.0/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c
    """  # noqa
    def __init__(self,
                 blur='GaussianBlur',
                 blur_ksize=None,
                 frg_seg='MOG2',
                 mat_morph=['erode', 'dilate'],
                 mat_morph_kernel=None,
                 ):
        self.blur = blur
        self.blur_ksize = blur_ksize
        self.frg_seg = frg_seg
        self.mat_morph = mat_morph
        self.mat_morph_kernel = mat_morph_kernel

    def fit(self):
        """Check parameters and set pipeline. No training.

        Returns
        -------
        self : object
            Returns self.
        """

        if self.blur not in ('blur', 'GaussianBlur', None):
            raise ValueError('Unknown filter for blurring')

        if self.blur_ksize is None:
            self._blur_ksize = (3, 3)
        else:
           self._blur_ksize = self.blur_ksize

        if self.frg_seg == 'MOG2':
            self._frg_seg_mth = cv.createBackgroundSubtractorMOG2()
        elif self.frg_seg == 'KNN':
            self._frg_seg_mth = cv.createBackgroundSubtractorKNN()
        else:
            raise ValueError('Unknown method for background subtraction')

        for mat_morph_op in set(self.mat_morph):
            if mat_morph_op not in ('erode', 'dilate'):
                raise ValueError('Unknown method for mathematical morphology')

        if self.mat_morph_kernel is None:
            self._mat_morph_krnl = cv.getStructuringElement(
                cv.MORPH_RECT,
                (5, 5),
            )
        else:
            self._mat_morph_krnl = self.mat_morph_kernel

        return self

    def predict(self, X):
        """Compute closest neighbor according to riemannian_metric.

        Parameters
        ----------
        X : ndarray of int, shape (n_height, n_width, n_color)
            Input frame.

        Returns
        -------
        y : int
            Prediction of intrusion: 1 if intrusion detected, 0 otherwise.
        """
        # denoising by blurring
        if self.blur:
            X = eval(f'cv.{self.blur}')(
                X,
                self._blur_ksize,
                cv.BORDER_DEFAULT,
            )

        # foreground segmentation
        mask_ = self._frg_seg_mth.apply(X)

        # denoising by mathematical morphology
        for mat_morph_op in self.mat_morph:
            mask_ = eval(f'cv.{mat_morph_op}')(mask_, self._mat_morph_krnl)

        self.motion_mask_ = mask_

        # intrusion detection
        if np.any(self.motion_mask_):
            return 1
        else:
            return 0
