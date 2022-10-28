# PyGOFPID

[![Code PythonVersion](https://img.shields.io/badge/python-3.8+-blue)](https://img.shields.io/badge/python-3.7+-blue)
[![License](https://img.shields.io/badge/licence-BSD--3--Clause-green)](https://img.shields.io/badge/license-BSD--3--Clause-green)

PyGOFPID is a Python package implementing
[perimeter intrusion detection (PID)](https://www.mdpi.com/1424-8220/22/9/3601)
systems for video protection.

Leveraging [OpenCV](https://github.com/opencv/opencv-python), you can easily
define a pipeline using good old fashioned (GOF) computer vision methods, like
[foreground detection](https://en.wikipedia.org/wiki/Foreground_detection)
and [mathematical morphology](https://en.wikipedia.org/wiki/Mathematical_morphology).

PyGOFPID is distributed under the open source 3-clause BSD license.

## Description

Class `GOFPID` is a versatile tool allowing to build a parameterizable PID
pipeline, composed of:

1. input frame denoising by spatial blurring;
2. foreground detection by background subtraction or frame differencing;
3. foreground mask denoising by mathematical morphology;
4. foreground blob creation;
5. blob tracking (WIP);
6. post-filtering;
7. intrusion detection.

See `examples/video.py` for a simple example.

## Comparison

GOFPID is useful for comparison between ancient and new generations of PID,
between good old fashioned and deep learning based PID.

For a benchmark between PIDs on [i-LIDS dataset](https://ieeexplore.ieee.org/document/4105319),
see [pyPID](https://gitlab.liris.cnrs.fr/dlohani/pypid).

See `examples/dataset_ilids.py` for configurations of i-LIDS views.

## Installation

#### From sources

To install PyGOFPID as a standard module:
```shell 
pip install path/to/PyGOFPID
```

To install PyGOFPID in editable / development mode, in the folder:
```shell
pip install poetry
poetry install
```

## Testing

Use `pytest`.

