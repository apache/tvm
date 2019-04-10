import tvm
import ctypes
import json
import os.path as osp
from sys import platform

def get_build_path():
    curr_path = osp.dirname(osp.abspath(osp.expanduser(__file__)))
    cfg = json.load(open(osp.join(curr_path, 'config.json')))
    return osp.join(curr_path, "..", "..", cfg['BUILD_NAME'])

def get_lib_ext():
    if platform == "darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    return ext

def get_lib_path(name):
    build_path = get_build_path()
    ext = get_lib_ext()
    libname = name + ext
    return osp.join(build_path, libname)

def _load_driver_lib():
    lib = get_lib_path("libdriver")
    try:
        return [ctypes.CDLL(lib, ctypes.RTLD_GLOBAL)]
    except OSError:
        return []

def load_driver():
    return tvm.get_global_func("tvm.vta.driver")

def load_vsim():
    lib = get_lib_path("libvsim")
    return tvm.module.load(lib, "vta-sim")

LIBS = _load_driver_lib()
