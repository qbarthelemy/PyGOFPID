# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['pygofpid']

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.7.0,<2.0.0', 'opencv-python>=3.4.0,<4.0.0.0']

setup_kwargs = {
    'name': 'pygofpid',
    'version': '1.0.1',
    'description': 'PyGOFPID is a Python package for good old fashioned perimeter intrusion detection system',
    'long_description': 'PyGOFPID is a Python package for good old fashioned perimeter intrusion detection system',
    'author': 'Quentin Barthelemy',
    'url': 'https://github.com/qbarthelemy/PyGOFPID',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
