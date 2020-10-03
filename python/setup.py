# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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

CURRENT_DIR = os.path.dirname(__file__)


DID_READ_LIBINFO = False


LIBINFO = None


def read_libinfo():
    global DID_READ_LIBINFO
    global LIBINFO
    if not DID_READ_LIBINFO:
        # We can not import `libinfo.py` in setup.py directly since __init__.py
        # Will be invoked which introduces dependences
        libinfo_py = os.path.join(CURRENT_DIR, "./tvm/_ffi/libinfo.py")
        LIBINFO = {"__file__": libinfo_py}
        exec(compile(open(libinfo_py, "rb").read(), libinfo_py, "exec"), LIBINFO, LIBINFO)
        DID_READ_LIBINFO = True


def get_version():
    read_libinfo()
    return LIBINFO["__version__"]


LIB_LIST = None


def get_lib_list():
    """Get list of paths to DSO libraries that should be included in an installation."""
    global LIB_LIST
    if LIB_LIST is None:
        read_libinfo()
        lib_path = LIBINFO["find_lib_path"]()
        LIB_LIST = [lib_path[0]]
        if LIB_LIST[0].find("runtime") == -1:
            for name in lib_path[1:]:
                if name.find("runtime") != -1:
                    LIB_LIST.append(name)
                    break

    return LIB_LIST


def config_cython():
    """Try to configure cython and return cython configuration"""
    if os.name == "nt":
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
        if os.name == "nt":
            library_dirs = ["tvm", "../build/Release", "../build"]
            libraries = ["libtvm"]
        else:
            library_dirs = None
            libraries = None
        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    "tvm._ffi.%s.%s" % (subdir, fn[:-4]),
                    ["tvm/_ffi/_cython/%s" % fn],
                    include_dirs=[
                        "../include/",
                        "../3rdparty/dmlc-core/include",
                        "../3rdparty/dlpack/include",
                    ],
                    extra_compile_args=["-std=c++14"],
                    library_dirs=library_dirs,
                    libraries=libraries,
                    language="c++",
                )
            )
        return cythonize(ret, compiler_directives={"language_level": 3})
    except ImportError:
        print("WARNING: Cython is not installed, will compile without cython module")
        return []


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


# For bdist_wheel only
include_libs = False
wheel_include_libs = False
if not os.getenv("CONDA_BUILD"):
    if "bdist_wheel" in sys.argv:
        wheel_include_libs = True
    else:
        include_libs = True

setup_kwargs = {}
if wheel_include_libs:
    with open("MANIFEST.in", "w") as fo:
        for path in get_lib_list():
            shutil.copy(path, os.path.join(CURRENT_DIR, "tvm"))
            _, libname = os.path.split(path)
            fo.write("include tvm/%s\n" % libname)
    setup_kwargs["include_package_data"] = True

if include_libs:
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    get_lib_list()
    for i, path in enumerate(LIB_LIST):
        LIB_LIST[i] = os.path.relpath(path, curr_path)
    setup_kwargs["include_package_data"] = True
    setup_kwargs["data_files"] = [("tvm", LIB_LIST)]


def get_package_data_files():
    # Relay standard libraries
    return ["relay/std/prelude.rly", "relay/std/core.rly"]


setup(
    name="tvm",
    version=get_version(),
    description="TVM: An End to End Tensor IR/DSL Stack for Deep Learning Systems",
    zip_safe=False,
    entry_points={"console_scripts": ["tvmc = tvm.driver.tvmc.main:main"]},
    install_requires=[
        "numpy",
        "scipy",
        "decorator",
        "attrs",
        "psutil",
        "typed_ast",
    ],
    extras_require={
        "test": ["pillow<7", "matplotlib"],
        "extra_feature": [
            "tornado",
            "psutil",
            "xgboost>=1.1.0",
            "mypy",
            "orderedset",
        ],
        "tvmc": [
            "tensorflow>=2.1.0",
            "tflite>=2.1.0",
            "onnx>=1.7.0",
            "onnxruntime>=1.0.0",
            "torch>=1.4.0",
            "torchvision>=0.5.0",
        ],
    },
    packages=find_packages(),
    package_dir={"tvm": "tvm"},
    package_data={"tvm": get_package_data_files()},
    distclass=BinaryDistribution,
    url="https://github.com/apache/incubator-tvm",
    ext_modules=config_cython(),
    **setup_kwargs
)


if wheel_include_libs:
    # Wheel cleanup
    os.remove("MANIFEST.in")
    for path in get_lib_list():
        _, libname = os.path.split(path)
        os.remove("tvm/%s" % libname)
