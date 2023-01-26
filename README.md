# PyGOFPID

[![Code PythonVersion](https://img.shields.io/badge/python-3.8+-blue)](https://img.shields.io/badge/python-3.7+-blue)
[![License](https://img.shields.io/badge/licence-BSD--3--Clause-green)](https://img.shields.io/badge/license-BSD--3--Clause-green)

PyGOFPID is a Python package implementing
[perimeter intrusion detection (PID)](https://www.mdpi.com/1424-8220/22/9/3601)
systems for video protection.

Leveraging [OpenCV](https://github.com/opencv/opencv-python), you can easily
define a pipeline using good old fashioned (GOF) computer vision methods, like
[foreground detection](https://en.wikipedia.org/wiki/Foreground_detection) and
[mathematical morphology](https://en.wikipedia.org/wiki/Mathematical_morphology).

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

## Comparison on i-LIDS

GOFPID is useful for comparison between ancient and new generations of PID,
between good old fashioned and deep learning based PID.

[i-LIDS](https://ieeexplore.ieee.org/document/4105319) is a standard dataset
for testing video based PID.

See [pyPID](https://gitlab.liris.cnrs.fr/dlohani/pypid) for a benchmark between
PID systems on i-LIDS dataset, with results given for
[edge-level evaluation after 1s](https://www.mdpi.com/1424-8220/22/9/3601)
(precision, recall, F1-score):

| Methods | View 1 | View 2 |
| ------- | ------ | ------ |
| YOLOv2 + SORT [Nayak2019](https://ieeexplore.ieee.org/document/9117960) | 0.62 0.52 0.57 | 0.99 0.47 0.64 |
| 3DCAE + MZ [Lohani2022](https://ieeexplore.ieee.org/document/9897472) | 0.90 0.74 0.81 | 0.83 0.64 0.73 |
| GOFPID v1.0.0 | 0.69 0.61 0.65 | 0.80 0.59 0.68 |

See `examples/dataset_ilids.py` for GOFPID configurations of i-LIDS views.

## Installation

#### From sources

To install PyGOFPID as a standard module:
```shell 
pip install git+https://github.com/qbarthelemy/PyGOFPID
```

To install PyGOFPID in editable / development mode, in the folder:
```shell
pip install poetry
poetry install
```

## Testing

Use `pytest`.

