# PyGOFPID

[![Code PythonVersion](https://img.shields.io/badge/python-3.8+-blue)](https://img.shields.io/badge/python-3.7+-blue)
[![License](https://img.shields.io/badge/licence-BSD--3--Clause-green)](https://img.shields.io/badge/license-BSD--3--Clause-green)

PyGOFPID is a Python package implementing
[perimeter intrusion detection (PID)](https://www.mdpi.com/1424-8220/22/9/3601)
systems for video protection.

Using [OpenCV](https://github.com/opencv/opencv-python), a Python pipeline uses
good old fashioned (GOF) computer vision methods, like
[background subtraction](https://en.wikipedia.org/wiki/Foreground_detection#Background_subtraction)
and [mathematical morphology](https://en.wikipedia.org/wiki/Mathematical_morphology).
It is useful for comparison with new generation of PID based on deep learning.

PyGOFPID is distributed under the open source 3-clause BSD license.

## Description

Class `GOFPID` is a versatile tool allowing to build a parameterizable PID
pipeline, composed of:

1. input frame denoising by spatial blurring;
2. foreground detection by background subtraction;
3. motion mask denoising by mathematical morphology;
4. motion blob tracking (WIP);
5. intrusion detection.

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

