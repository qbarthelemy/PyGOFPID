from setuptools import find_packages, setup

setup(
    name='pygofpid',
    version='1.2.0',
    description='Python package for good old fashioned perimeter intrusion detection systems for video protection',
    long_description='PyGOFPID is a Python package for good old fashioned perimeter intrusion detection systems for video protection',
    author='Quentin Barthelemy',
    url='https://github.com/qbarthelemy/PyGOFPID',
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0.0",
        "opencv-python>=4.11",
    ],
    extras_require={
        "tests": [
            "flake8",
            "pytest",
        ],
    },
)
