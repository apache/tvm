#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup tvm package."""
from __future__ import absolute_import
import os
import sys
import setuptools

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'tvm/_ffi/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, 'rb').read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
print(LIB_PATH)
__version__ = libinfo['__version__']

def config_cython():
    """Try to configure cython and return cython configuration"""
    if os.name == 'nt':
        print("WARNING: Cython is not supported on Windows, will compile without cython module")
        return []
    try:
        from Cython.Build import cythonize
        # from setuptools.extension import Extension
        if sys.version_info >= (3, 0):
            subdir = "_cy3"
        else:
            subdir = "_cy2"
        ret = []
        path = "tvm/_ffi/_cython"
        if os.name == 'nt':
            library_dirs = ['tvm', '../build/Release', '../build']
            libraries = ['libtvm']
        else:
            library_dirs = None
            libraries = None
        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "tvm._ffi.%s.%s" % (subdir, fn[:-4]),
                ["tvm/_ffi/_cython/%s" % fn],
                include_dirs=["../include/",
                              "../dmlc-core/include",
                              "../dlpack/include",
                ],
                library_dirs=library_dirs,
                libraries=libraries,
                language="c++"))
        return cythonize(ret)
    except ImportError:
        print("WARNING: Cython is not installed, will compile without cython module")
        return []

setuptools.setup(
    name='tvm',
    version=__version__,
    description='A domain specific language(DSL) for tensor computations.',
    install_requires=[
        'numpy',
        ],
    zip_safe=False,
    packages=[
        'tvm._ffi', 'tvm._ffi._ctypes',
        'tvm._ffi._cy2', 'tvm._ffi._cy3'
    ],
    data_files=[('tvm', [LIB_PATH[0]])],
    url='https://github.com/tqchen/tvm',
    ext_modules=config_cython())
