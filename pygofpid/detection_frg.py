"""Methods for foreground detection."""

import cv2 as cv
import numpy as np

from .helpers import dist_euclidean

BACKGROUND = 0
FOREGROUND = 255


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
        self._model = None

    def apply(self, X):
        """Estimate foreground.

        Parameters
        ----------
        X : ndarray of int, shape (n_height, n_width) or \
                (n_height, n_width, n_channel)
            Input frame.

        Returns
        -------
        F : ndarray, shape (n_height, n_width)
            Foreground frame.
        """

        if self._model is None:
            F = np.zeros_like(X)
        else:
            diff = cv.absdiff(X, self._model)
            _, F = cv.threshold(
                diff,
                self.threshold,
                FOREGROUND,
                cv.THRESH_BINARY,
            )

        self._model = X
        if F.ndim > 2:
            F = np.mean(F, axis=-1, dtype=np.uint8, keepdims=False)

        return F


class ViBe():
    """Foreground detection by VIsual Background Extractor (ViBe).

    Parameters
    ----------
    n_samples_per_pixel : int, default=20
        Number of samples per pixel.

    sphere_radius : float, default=20
        Radius of the sphere.

    n_samples_close : int, default=2
        Number of close samples for being part of the background.

    subsampling_factor : int, default=16
        Amount of random subsampling.

    seed : {None, int, array_like}, default=42
        Random seed used to initialize the pseudo-random number generator.

    References
    ----------
    .. [1] ViBe: A universal background subtraction algorithm for video
        sequences
        Barnich, O., & Van Droogenbroeck, M.
        IEEE Trans Image Process, 2010.
    """

    def __init__(
        self,
        n_samples_per_pixel=20,
        sphere_radius=20,
        n_samples_close=2,
        subsampling_factor=16,
        seed=42,
    ):
        self.n_samples_per_pixel = n_samples_per_pixel
        self.sphere_radius = sphere_radius
        self.n_samples_close = n_samples_close
        self.subsampling_factor = subsampling_factor
        self.seed = seed
        self._model = None
        self._rnd = None

    def apply(self, X):
        """Estimate foreground.

        Parameters
        ----------
        X : ndarray of int, shape (n_height, n_width) or \
                (n_height, n_width, n_channel)
            Input frame.

        Returns
        -------
        F : ndarray of int, shape (n_height, n_width)
            Foreground frame.
        """
        n_height, n_width = X.shape[:2]
        if self._model is None:
            self._model = np.zeros((*X.shape, self.n_samples_per_pixel))
        F = np.zeros((n_height, n_width), dtype=np.uint8)

        if self._rnd == None:
            self._rnd = np.random.RandomState(seed=self.seed)

        for y in range(n_height):
            for x in range(n_width):

                # compare pixel to background model
                c, i = 0, 0
                while c < self.n_samples_close and i < self.n_samples_per_pixel:
                    dist = dist_euclidean(X[y, x], self._model[y, x, ..., i])
                    if dist < self.sphere_radius:
                        c += 1
                    i += 1

                # classify pixel and update model
                if c >= self.n_samples_close:
                    F[y, x] = BACKGROUND

                    # update current pixel model
                    if self._get_random_int(0, self.subsampling_factor - 1) == 0:
                        r = self._get_random_int(0, self.n_samples_per_pixel - 1)
                        self._model[y, x, ..., r] = X[y, x]

                    # update neighboring pixel model
                    if self._get_random_int(0, self.subsampling_factor - 1) == 0:
                        yn = self._get_random_coord(y, n_height)
                        xn = self._get_random_coord(x, n_width)
                        r = self._get_random_int(0, self.n_samples_per_pixel - 1)
                        self._model[yn, xn, ..., r] = X[y, x]

                else:
                    F[y, x] = FOREGROUND

        return F

    def _get_random_int(self, low, high):
        return self._rnd.randint(low, high)

    def _get_random_coord(self, val, val_max):
        """Return a random coord in the 8-connected neighborhood."""
        delta = self._rnd.choice([-1, 0, 1])
        val_new = min(max(val + delta, 0), val_max - 1)
        return val_new
