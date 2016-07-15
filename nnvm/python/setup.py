from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(
    name='nnvm',
    ext_modules = cythonize([
        Extension("nnvm/symbolx",
                  ["nnvm/symbolx.pyx"],
                  libraries=["nnvm"],
                  language="c++")
    ])
)
