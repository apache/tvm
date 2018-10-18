from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup (
        name = 'helloworld',
        ext_modules = cythonize([Extension("helloworld", ["helloworld.pyx"])]),
)
