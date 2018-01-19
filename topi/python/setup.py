# pylint: disable=invalid-name, exec-used
"""Setup TOPI package."""
from __future__ import absolute_import
import sys
import os

from setuptools import find_packages
from setuptools.dist import Distribution

if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

def get_lib_names():
    if sys.platform.startswith('win32'):
        return ['libtvm_topi.dll', 'tvm_topi.dll']
    if sys.platform.startswith('darwin'):
        return ['libtvm_topi.dylib', 'tvm_topi.dylib']
    return ['libtvm_topi.so', 'tvm_topi.so']

def get_lib_path():
    """Get library path, name and version"""
    # We can not import `libinfo.py` in setup.py directly since __init__.py
    # Will be invoked which introduces dependences
    CURRENT_DIR = os.path.dirname(__file__)
    libinfo_py = os.path.join(CURRENT_DIR, '../../python/tvm/_ffi/libinfo.py')
    libinfo = {'__file__': libinfo_py}
    exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)
    lib_path = libinfo['find_lib_path'](get_lib_names())
    version = libinfo['__version__']
    libs = [lib_path[0]]
    if libs[0].find("runtime") == -1:
        for name in lib_path[1:]:
            if name.find("runtime") != -1:
                libs.append(name)
                break
    return libs, version

LIB_LIST, __version__ = get_lib_path()

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
for i, path in enumerate(LIB_LIST):
    LIB_LIST[i] = os.path.relpath(path, curr_path)
setup_kwargs = {
    "include_package_data": True,
    "data_files": [('topi', LIB_LIST)]
}

setup(name='topi',
      version=__version__,
      description="TOPI: TVM operator index",
      install_requires=[
        "numpy",
        "decorator",
        ],
      packages=find_packages(),
      url='https://github.com/dmlc/tvm',
      **setup_kwargs)
