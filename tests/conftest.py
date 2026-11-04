import numpy as np
import pytest


@pytest.fixture
def get_video():
    """Return a function generating a video of a square crossing the frame."""

    def _get_video(
        size=(48, 64),
        n_frames_bkg=30,
        square_size=12,
        speed=2,
        with_square=True,
        seed=17,
    ):
        """Generate a video where a bright square crosses from left to right.

        The background is uniform with a slight Gaussian noise. The first
        frames contain only background, then the square crosses the frame at
        mid-height, from the left border to the right border.

        Parameters
        ----------
        size : tuple, default=(48, 64)
            Shape of frames: (n_height, n_width) or (n_height, n_width, n_ch).
        n_frames_bkg : int, default=30
            Number of frames containing only background. With too few frames,
            MOG2 learns quickly and absorbs the square into the background.
        square_size : int, default=12
            Size of the square, in pixels.
        speed : int, default=2
            Horizontal displacement of the square between two frames, in
            pixels.
        with_square : bool, default=True
            If False, the square is not drawn, giving a background video.
        seed : int, default=17
            Random seed for the background noise.

        Returns
        -------
        frames : list of ndarray of uint8, shape size
            Frames of the video.
        boxes : list of None or tuple
            For each frame, None if there is no square, or the box of the
            square (x_min, y_min, x_max, y_max), with exclusive max.
        """
        rnd = np.random.default_rng(seed)
        n_height, n_width = size[:2]
        y_min = (n_height - square_size) // 2
        x_mins = range(0, n_width - square_size + 1, speed)

        frames, boxes = [], []
        for i in range(n_frames_bkg + len(x_mins)):
            noise = rnd.normal(scale=1, size=size)
            frame = np.clip(np.rint(100 + noise), 0, 255).astype(np.uint8)
            box = None
            if with_square and i >= n_frames_bkg:
                x_min = x_mins[i - n_frames_bkg]
                box = (x_min, y_min, x_min + square_size, y_min + square_size)
                frame[box[1]:box[3], box[0]:box[2]] = 220
            frames.append(frame)
            boxes.append(box)

        return frames, boxes

    return _get_video
