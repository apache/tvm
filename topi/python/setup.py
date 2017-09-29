# pylint: disable=invalid-name, exec-used
"""Setup TOPI package."""
from __future__ import absolute_import
import sys

from setuptools import find_packages
from setuptools.dist import Distribution

if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

__version__ = "0.1.0"

setup(name='topi',
      version=__version__,
      description="TOPI: TVM operator index",
      install_requires=[
        "numpy",
        "decorator",
        ],
      packages=find_packages(),
      url='https://github.com/dmlc/tvm')
