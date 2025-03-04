from setuptools import find_packages, setup

setup(
    name='pygofpid',
    version='1.0.4',
    description='Python package for good old fashioned perimeter intrusion detection systems for video protection',
    long_description='PyGOFPID is a Python package for good old fashioned perimeter intrusion detection systems for video protection',
    author='Quentin Barthelemy',
    url='https://github.com/qbarthelemy/PyGOFPID',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.7.0,<2.0.0",
        "opencv-python>=3.4.0,<4.0.0.0",
    ],
    python_requires='>=3.8,<4.0',
)
