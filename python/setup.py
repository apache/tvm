# pylint: disable=invalid-name, exec-used
"""Setup TVM package."""
from __future__ import absolute_import
import os
import shutil
import sys
import sysconfig
import platform

from setuptools import find_packages
from setuptools.dist import Distribution

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

# We can not import `libinfo.py` in setup.py directly since __init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, './tvm/_ffi/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
_, LIB_NAME = os.path.split(LIB_PATH[0])
__version__ = libinfo['__version__']

def config_cython():
    """Try to configure cython and return cython configuration"""
    if os.name == 'nt':
        print("WARNING: Cython is not supported on Windows, will compile without cython module")
        return []
    sys_cflags = sysconfig.get_config_var("CFLAGS")

    if "i386" in sys_cflags and "x86_64" in sys_cflags:
        print("WARNING: Cython library may not be compiled correctly with both i386 and x64")
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

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

# For bdist_wheel only
if "bdist_wheel" in sys.argv:
    shutil.copy(LIB_PATH[0], os.path.join(CURRENT_DIR, 'tvm'))
    with open("MANIFEST.in", "w") as fo:
        fo.write("include tvm/%s\n" % LIB_NAME)
    setup_kwargs = {
        "include_package_data": True
    }
else:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    rpath = os.path.relpath(LIB_PATH[0], curr_path)
    setup_kwargs = {
        "include_package_data": True,
        "data_files": [('tvm', [rpath])]
    }

setup(name='tvm',
      version=__version__,
      description="TVM: An End to End Tensor IR/DSL Stack for Deep Learning Systems",
      zip_safe=False,
      install_requires=[
        'numpy',
        'decorator',
        ],
      packages=find_packages(),
      distclass=BinaryDistribution,
      url='https://github.com/dmlc/tvm',
      ext_modules=config_cython(),
      **setup_kwargs)

# Wheel cleanup
if "bdist_wheel" in sys.argv:
    os.remove("MANIFEST.in")
    os.remove("tvm/%s" % LIB_NAME)
