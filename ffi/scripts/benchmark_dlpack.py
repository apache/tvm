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
"""
This script is used to benchmark the API overhead of different
python FFI API calling overhead, through DLPack API.

Specifically, we would like to understand the overall overhead
python/C++ API calls. The general goal is to understand the overall
space and get a sense of what are the possible operations.

We pick function f(x, y, z) where x, y, z are length 1 tensors.
The benchmark is running in eager mode so we can see what is possible.
It is orthogonal to other optimizations. For example cudagraph can
eliminate these overheads completely. So the goal is to get a sense
of what is possible under eager mode.

Summary of some takeaways:
- numpy.add roughly takes 0.36 us per call, which gives roughly what can
  be done in python env.
- torch.add on gpu takes about 3.7us per call, giving us an idea of what
  roughly we need to get to in eager mode.
-

"""
import torch
import numpy as np
from tvm import ffi as tvm_ffi
import time


def print_speed(name, speed):
    print(f"{name:<40} {speed} sec/call")


def print_error(name, error):
    print(f"{name:<40} {error}")


def baseline_torch_add(repeat):
    """Run torch.add with one element"""

    def run_bench(device):
        x = torch.arange(1, device=device)
        y = torch.arange(1, device=device)
        z = torch.arange(1, device=device)

        torch.add(x, y, out=z)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for i in range(repeat):
            torch.add(x, y, out=z)
        # note we deliberately do not use torch.cuda.synchronize()
        # because we want to see the overhead of the FFI call.
        end = time.time()
        print_speed(f"torch.add[{device}]", (end - start) / repeat)

    # rough take away: add on cuda roughly takes 3e-6 sec/call
    run_bench("cpu")
    run_bench("cuda")


def baseline_numpy_add(repeat):
    """Run numpy.add with one element"""
    x = np.arange(1)
    y = np.arange(1)
    z = np.arange(1)

    np.add(x, y, out=z)
    start = time.time()
    for i in range(repeat):
        np.add(x, y, out=z)
    end = time.time()
    speed = (end - start) / repeat
    print_speed("numpy.add", speed)


def baseline_cupy_add(repeat):
    """Run cupy.add with one element"""
    try:
        import cupy
    except ImportError:
        # skip if cupy is not installed
        return
    x = cupy.arange(1)
    y = cupy.arange(1)
    z = cupy.arange(1)

    cupy.add(x, y, out=z)
    start = time.time()
    for i in range(repeat):
        cupy.add(x, y, out=z)
    end = time.time()
    speed = (end - start) / repeat
    print_speed("cupy.add", speed)


def tvm_ffi_nop(repeat):
    """Overhead of tvm FFI python call via calling a NOP.

    testing.nop is defined in c++ and do nothing.
    """
    nop = tvm_ffi.get_global_func("testing.nop")
    x = tvm_ffi.from_dlpack(torch.arange(1))
    y = tvm_ffi.from_dlpack(torch.arange(1))
    z = tvm_ffi.from_dlpack(torch.arange(1))
    nop(x, y, z)
    start = time.time()
    for i in range(repeat):
        y = tvm_ffi.from_dlpack(x)
    end = time.time()
    print_speed("tvm.ffi.nop", (end - start) / repeat)


def bench_ffi_nop_from_dlpack(name, x, y, z, repeat):
    """run dlpack conversion + tvm.ffi.nop

    Measures overhead of running dlpack for each args then invoke
    """
    nop = tvm_ffi.get_global_func("testing.nop")
    tx = tvm_ffi.from_dlpack(x)
    ty = tvm_ffi.from_dlpack(y)
    tz = tvm_ffi.from_dlpack(z)
    nop(tx, ty, tz)

    start = time.time()
    for i in range(repeat):
        tx = tvm_ffi.from_dlpack(x)
        ty = tvm_ffi.from_dlpack(y)
        tz = tvm_ffi.from_dlpack(z)
        nop(tx, ty, tz)
    end = time.time()
    print_speed(name, (end - start) / repeat)


def tvm_ffi_nop_from_torch_dlpack(repeat):
    """run dlpack conversion + tvm.ffi.nop

    Measures overhead of running dlpack for each args then invoke
    """
    x = torch.arange(1)
    y = torch.arange(1)
    z = torch.arange(1)
    bench_ffi_nop_from_dlpack("tvm.ffi.nop+from_dlpack(torch)", x, y, z, repeat)


def tvm_ffi_nop_from_numpy_dlpack(repeat):
    """run dlpack conversion + tvm.ffi.nop

    Measures overhead of running dlpack for each args then invoke
    """
    x = np.arange(1)
    y = np.arange(1)
    z = np.arange(1)
    bench_ffi_nop_from_dlpack("tvm.ffi.nop+from_dlpack(numpy)", x, y, z, repeat)


def tvm_ffi_self_dlpack_nop(repeat):
    """run dlpack conversion + tvm.ffi.nop

    Measures overhead of running dlpack for each args then invoke
    """
    x = tvm_ffi.from_dlpack(torch.arange(1))
    y = tvm_ffi.from_dlpack(torch.arange(1))
    z = tvm_ffi.from_dlpack(torch.arange(1))
    bench_ffi_nop_from_dlpack("tvm.ffi.nop+from_dlpack(tvm)", x, y, z, repeat)


def bench_ffi_nop_from_dlpack(name, x, y, z, repeat):
    """run dlpack conversion + tvm.ffi.nop

    Measures overhead of running dlpack for each args then invoke
    """
    nop = tvm_ffi.get_global_func("testing.nop")
    tx = tvm_ffi.from_dlpack(x)
    ty = tvm_ffi.from_dlpack(y)
    tz = tvm_ffi.from_dlpack(z)
    nop(tx, ty, tz)

    start = time.time()
    for i in range(repeat):
        tx = tvm_ffi.from_dlpack(x)
        ty = tvm_ffi.from_dlpack(y)
        tz = tvm_ffi.from_dlpack(z)
        nop(tx, ty, tz)
    end = time.time()
    print_speed(name, (end - start) / repeat)


def tvm_ffi_nop_from_torch_utils_to_dlpack(repeat):
    """
    Measures overhead of running dlpack for each args then invoke
    but uses the legacy torch.utils.dlpack.to_dlpack API

    This helps to measure possible implementation overhead of torch.
    """
    nop = tvm_ffi.get_global_func("testing.nop")
    x = torch.arange(1)
    y = torch.arange(1)
    z = torch.arange(1)

    tx = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    ty = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(y))
    tz = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(z))
    nop(tx, ty, tz)

    start = time.time()
    for i in range(repeat):
        tx = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        ty = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(y))
        tz = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(z))
        nop(tx, ty, tz)
    end = time.time()
    speed = (end - start) / repeat
    print_speed("tvm.ffi.nop+from_dlpack(torch.utils)", speed)


def bench_tvm_ffi_nop_autodlpack(name, x, y, z, repeat):
    """
    Measures overhead of running dlpack via auto convert by directly
    take torch.Tensor as inputs.
    """
    nop = tvm_ffi.get_global_func("testing.nop")
    nop(x, y, z)
    start = time.time()
    for i in range(repeat):
        nop(x, y, z)
    end = time.time()
    speed = (end - start) / repeat
    print_speed(name, speed)


def tvm_ffi_nop_autodlpack_from_torch(repeat, device="cpu"):
    """
    Measures overhead of running dlpack via auto convert by directly
    take torch.Tensor as inputs.
    """
    # use larger to ensure alignment req is met
    x = torch.arange(1, device=device)
    y = torch.arange(1, device=device)
    z = torch.arange(1, device=device)
    bench_tvm_ffi_nop_autodlpack(f"tvm.ffi.nop.autodlpack(torch[{device}])", x, y, z, repeat)


def tvm_ffi_nop_autodlpack_from_numpy(repeat):
    """
    Measures overhead of running dlpack via auto convert by directly
    take numpy.ndarray as inputs.
    """
    # use larger to ensure alignment req is met
    x = np.arange(256)
    y = np.arange(256)
    z = np.arange(256)
    bench_tvm_ffi_nop_autodlpack("tvm.ffi.nop.autodlpack(numpy)", x, y, z, repeat)


def bench_to_dlpack(x, name, repeat):
    x.__dlpack__()
    start = time.time()
    for i in range(repeat):
        x.__dlpack__()
    end = time.time()
    speed = (end - start) / repeat
    print_speed(name, speed)


def bench_to_dlpack_versioned(x, name, repeat, max_version=(1, 1)):
    """
    Measures overhead of running dlpack with latest 1.1.
    """
    try:
        x.__dlpack__(max_version=max_version)
        start = time.time()
        for i in range(repeat):
            x.__dlpack__(max_version=max_version)
        end = time.time()
        speed = (end - start) / repeat
        print_speed(name, speed)
    except Exception as e:
        print_error(name, e)


def bench_torch_utils_to_dlpack(repeat):
    """
    Measures overhead of running torch.utils.dlpack.to_dlpack
    """
    x = torch.arange(1)
    torch.utils.dlpack.to_dlpack(x)
    start = time.time()
    for i in range(repeat):
        torch.utils.dlpack.to_dlpack(x)
    end = time.time()
    speed = (end - start) / repeat
    print_speed("torch.utils.dlpack.to_dlpack", speed)


def main():
    repeat = 10000
    print("-----------------------------")
    print("Benchmark f(x, y, z) overhead")
    print("-----------------------------")
    baseline_numpy_add(repeat)
    baseline_torch_add(repeat)
    baseline_cupy_add(repeat)
    tvm_ffi_nop(repeat)
    tvm_ffi_nop_from_torch_dlpack(repeat)
    tvm_ffi_nop_from_numpy_dlpack(repeat)
    tvm_ffi_self_dlpack_nop(repeat)
    tvm_ffi_nop_from_torch_utils_to_dlpack(repeat)
    tvm_ffi_nop_autodlpack_from_torch(repeat, "cpu")
    tvm_ffi_nop_autodlpack_from_torch(repeat, "cuda")
    tvm_ffi_nop_autodlpack_from_numpy(repeat)
    print("-------------------------------")
    print("Benchmark x.__dlpack__ overhead")
    print("-------------------------------")
    bench_torch_utils_to_dlpack(repeat)
    bench_to_dlpack(torch.arange(1), "torch.__dlpack__", repeat)
    bench_to_dlpack(np.arange(1), "numpy.__dlpack__", repeat)
    bench_to_dlpack(tvm_ffi.from_dlpack(torch.arange(1)), "tvm.__dlpack__", repeat)
    print("---------------------------------------------------")
    print("Benchmark x.__dlpack__(max_version=(1,1)) overhead")
    print("---------------------------------------------------")
    bench_to_dlpack_versioned(torch.arange(1), "torch.__dlpack__(max_version=(1,1))", repeat)
    bench_to_dlpack_versioned(np.arange(1), "numpy.__dlpack__(max_version=(1,1))", repeat)
    bench_to_dlpack_versioned(
        tvm_ffi.from_dlpack(torch.arange(1)), "tvm.__dlpack__(max_version=(1,1))", repeat
    )


if __name__ == "__main__":
    main()
