#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup tvm package."""
from __future__ import absolute_import
import os
import sys
import setuptools

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'tvm/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, 'rb').read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']

setuptools.setup(
    name='tvm',
    version=__version__,
    description='A domain specific language(DSL) for tensor computations.',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        ],
    data_files=[('tvm', [LIB_PATH[0]])]
    )
