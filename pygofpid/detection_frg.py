"""Methods."""

import cv2 as cv
import numpy as np



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


