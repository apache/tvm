# pylint: disable=invalid-name, exec-used
"""Setup TVM package."""
from __future__ import absolute_import
import os
import shutil
import sys
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

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, '../../python/tvm/_ffi/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']

def config_cython():
    """Try to configure cython and return cython configuration"""
    if os.name == 'nt':
        print("WARNING: Cython is not supported on Windows, will compile without cython module")
        return []
    if platform.system() == 'Darwin':
        print("WARNING: Cython is not linking with correct arch on Mac OSX, will compile without cython module")
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
                include_dirs=["../../include/",
                              "../../dmlc-core/include",
                              "../../dlpack/include",
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

# Quick hack
LIBS = os.listdir("../../lib/")
shutil.rmtree(os.path.join(CURRENT_DIR, 'tvm'), ignore_errors=True)
shutil.copytree(os.path.join(CURRENT_DIR, '../../python/tvm'),
                os.path.join(CURRENT_DIR, 'tvm'))
fo = open("MANIFEST.in", "w")
for lib in LIBS:
    shutil.copy(os.path.join("../../lib/", lib), os.path.join(CURRENT_DIR, 'tvm'))
    fo.write("include tvm/%s\n" % lib)
fo.close()

setup(name='tvm',
      version=__version__,
      description='A domain specific language(DSL) for tensor computations.',
      zip_safe=False,
      install_requires=[
        'numpy',
        'decorator',
        ],
      packages=find_packages(),
      include_package_data=True,
      package_data={'tvm': [os.path.join('tvm', lib) for lib in LIBS]},
      #data_files=[('tvm',[os.path.join('tvm', lib) for lib in LIBS])],
      distclass=BinaryDistribution,
      url='https://github.com/dmlc/tvm',
      ext_modules=config_cython())
