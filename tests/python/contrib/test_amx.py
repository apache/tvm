import pytest
import itertools
import numpy as np
import os
import sys
import subprocess
import math
import collections

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl
import tvm.testing


def test_amx_tensorize(dtypt="int8"):
    pass

def test_amx_check_support():
    amx_init = tvm.get_global_func("runtime.amx_init")
    amx_tileconfig = tvm.get_global_func("runtime.amx_tileconfig")
    if not amx_init():
        print("[ ERROR ] AMX not inited !!!!!")
    if not amx_tileconfig(16, 64):
        print("[ ERROR ] AMX not tile configed !!!!!")


if __name__ == "__main__":
    print("[Test for TVM - LLVM - AMX intrinsic call pid: {}]".format(os.getpid()) )
    test_amx_check_support()
