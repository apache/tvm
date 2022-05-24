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
# pylint: disable=invalid-name
"""measure bandwidth and compute peak"""

import logging
import tvm
from tvm import te
from tvm.target import Target
from . import utils
from .. import rpc


def _convert_to_remote(func, remote):
    """convert module function to remote rpc function"""
    temp = utils.tempdir()
    path_dso = temp.relpath("tmp_func.tar")
    func.export_library(path_dso)

    remote.upload(path_dso)
    func = remote.load_module("tmp_func.tar")
    return func


def measure_bandwidth_sum(
    total_item,
    item_per_thread,
    stride,
    base_type,
    bits,
    lanes,
    target,
    target_host,
    remote,
    dev,
    n_times,
):
    """measure memory bandwidth of gpu by product reduction for a given type

    The IR for measurement is

    for each thread
        for i in 1..num_per_thread:
            y[global_id] = y[global_id] * x[base + i * stride]

    Parameters
    ----------
    total_item: int
        number of elements in input array
    item_per_thread: int
        number of elements each thread accumulates
    stride: int
        stride in memory access
    base_type: str
        can be "int", "float"
    bits: int
        can be 16, 32
    lanes: int
       lane of the vector type, can be 1, 2, 4, 8, 16
    target: :any:`tvm.target.Target`
        the target and option of the compilation.
    target_host : str or :any:`tvm.target.Target`
        host compilation target
    dev: Device
        the device of array
    remote: tvm.rpc.RPCSession
        remote rpc session
    n_times: int
        number of runs for taking mean

    Returns
    -------
    GBPS: float
         gigabyte per second
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    n, m = total_item, item_per_thread
    n //= lanes

    base_type = str(base_type) + str(bits)
    dtype = base_type if lanes == 1 else base_type + "x" + str(lanes)

    k = te.reduce_axis((0, m), name="k")

    x = te.placeholder((n,), dtype=dtype, name="x")
    op = te.comm_reducer(lambda x, y: x * y, lambda t: tvm.tir.const(1, dtype=t), name="sum")
    y = te.compute(
        (n // m,), lambda i: op(x[i // stride * stride * m + i % stride + k * stride], axis=k)
    )
    s = te.create_schedule(y.op)

    yo, yi = s[y].split(y.op.axis[0], target.max_num_threads)
    s[y].bind(yo, te.thread_axis("blockIdx.x"))
    s[y].bind(yi, te.thread_axis("threadIdx.x"))
    s[y].unroll(k)

    try:
        func = tvm.build(s, [x, y], target)

        x = tvm.nd.empty((n,), dtype=dtype, device=dev)
        y = tvm.nd.empty((n // m,), dtype=dtype, device=dev)

        func = _convert_to_remote(func, remote)
        time_f = func.time_evaluator(func.entry_name, dev, number=n_times)
        time = time_f(x, y).mean
    except tvm._ffi.base.TVMError:
        # build error (occur when device does not support half)
        return -1

    return 1.0 * (total_item * bits / 8) / 1e9 / time


def measure_bandwidth_all_types(
    total_item, item_per_thread, n_times, target, target_host, remote, dev, verbose=True
):
    """measure memory bandwidth for all types

    Parameters
    ----------
    total_item: int
        number of elements in input array
    item_per_thread: int
        number of elements each thread accmulates
    n_times: int
        number of runs for averaging
    target: :any:`tvm.target.Target`
        the target and option of the compilation.
    target_host : str or :any:`tvm.target.Target`
        host compilation target
    remote: tvm.rpc.RPCSession
        remote rpc session
    dev: Device
        the device of array
    verbose: bool
        whether outputs immediate result

    Returns
    -------
    result: list
        a list of (type_name, GBPS) pairs
    """
    target, target_host = Target.canon_target_and_host(target, target_host)
    max_threads = target.max_num_threads

    result = []
    for base_type in ["float"]:
        for bits in [32]:
            for lanes in [1, 2, 4, 8, 16]:
                max_speed = -1e9
                # try different strides
                for stride in [max_threads, total_item // (lanes * item_per_thread)]:
                    speed = measure_bandwidth_sum(
                        total_item,
                        item_per_thread,
                        stride,
                        base_type,
                        bits,
                        lanes,
                        target,
                        target_host,
                        remote,
                        dev,
                        n_times,
                    )
                    max_speed = max(max_speed, speed)
                type_name = base_type + str(bits)
                result.append(["%sx%d" % (type_name, lanes), max_speed])
                if verbose:
                    logging.info("\t%-10s %.2f GBPS", result[-1][0], result[-1][1])
    return result


def measure_compute_mad(
    total_item, item_per_thread, base_type, bits, lanes, target, target_host, remote, dev, n_times
):
    """measure peak compute speed by computing mad for a type

    The IR for measurement is

    for each thread
        for i in 1..item_per_thread
            x = mad(x, x, y)
            y = mad(y, y, x)

    Parameters
    ----------
    total_item: int
        number of elements in input array
    item_per_thread: int
        number of operations each thread does
    base_type: str
        can be "int", "float"
    bits: int
        can be 16, 32
    lanes: int
       lane of the vector type, can be 1, 2, 4, 8, 16
    target: :any:`tvm.target.Target`
        the target and option of the compilation.
    target_host : str or :any:`tvm.target.Target`
        host compilation target
    remote: tvm.rpc.RPCSession
        if it is not None, use remote rpc session
    dev: Device
        the device of array
    n_times: int
        number of runs for taking mean

    Returns
    -------
    GOPS: float
         giga operation per second
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    n = total_item

    if bits >= 64 or lanes >= 16:
        n //= 2

    max_threads = target.max_num_threads

    base_type = str(base_type) + str(bits)
    dtype = base_type if lanes == 1 else base_type + "x" + str(lanes)

    def extern(ins, outs):
        # pylint: disable=unused-argument
        """construct measurement function by building IR directly"""
        ib = tvm.tir.ir_builder.create()

        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")

        ib.scope_attr(bx, "thread_extent", n // max_threads)
        ib.scope_attr(tx, "thread_extent", max_threads)

        idx = bx.var * max_threads + tx.var

        a = ib.allocate(dtype, (1), name="a", scope="local")
        b = ib.allocate(dtype, (1), name="b", scope="local")

        a[0] = outs[0].vload(idx, dtype)
        b[0] = outs[0].vload(idx, dtype)

        if base_type.find("float") != -1:

            def mad_func(x, y):
                return x * x + y

        else:

            def mad_func(x, y):
                return y * y + x

        for _ in range(item_per_thread // 4 // lanes):
            a[0] = mad_func(a[0], b[0])
            b[0] = mad_func(b[0], a[0])

        ib.emit(outs[0].vstore(idx, b[0]))
        return ib.get()

    y = te.extern((n,), [], extern, name="y", dtype=dtype)
    s = te.create_schedule(y.op)

    try:
        func = tvm.build(s, [y], target)
        func = _convert_to_remote(func, remote)
        time_f = func.time_evaluator(func.entry_name, dev, number=n_times)
        y = tvm.nd.empty((n,), dtype=dtype, device=dev)
        time = time_f(y).mean
    except tvm._ffi.base.TVMError:
        # build error (occur when device does not support half)
        return -1

    return 1.0 * (n * item_per_thread) / 1e9 / time


def measure_compute_all_types(
    total_item, item_per_thread, n_times, target, target_host, remote, dev, verbose=True
):
    """measure peak flops for all types

    Parameters
    ----------
    total_item: int
        number of elements in input array
    item_per_thread: int
        number of elements each thread accmulates
    n_times: int
        number of runs for averaging
    target: :any:`tvm.target.Target`
        the target and option of the compilation.
    target_host : str or :any:`tvm.target.Target`
        host compilation target
    remote: tvm.rpc.RPCSession
        remote rpc session
    dev: Device
        the device of array
    verbose: bool
        whether outputs immediate result

    Returns
    -------
    result: list
        a list of (type_name, GFLOPS/GIOPS) pairs
    """
    target, target_host = Target.canon_target_and_host(target, target_host)

    result = []
    for base_type in ["float", "int"]:
        for bits in [16, 32, 64]:
            for lanes in [1, 2, 4, 8, 16]:
                if base_type == "int" and bits != 32:  # only measure int32
                    continue

                max_speed = -1e9
                for per_thread in [item_per_thread // 2, item_per_thread, item_per_thread * 2]:
                    speed = measure_compute_mad(
                        total_item,
                        per_thread,
                        base_type,
                        bits,
                        lanes,
                        target,
                        target_host,
                        remote,
                        dev,
                        n_times,
                    )
                    max_speed = max(max_speed, speed)
                type_name = base_type + str(bits)
                result.append(["%sx%d" % (type_name, lanes), max_speed])

                unit = "GFLOPS" if base_type == "float" else "GIOPS"

                if verbose:
                    logging.info("\t%-10s %.2f %s", result[-1][0], result[-1][1], unit)

    return result


def measure_peak_all(target, target_host, host, port):
    """measure memory bandwidth and peak compute for gpu devices

    Parameters
    ----------
    target: str or :any:`tvm.target.Target`
    target_host: str
    host: str
    port: int
    """

    target, target_host = Target.canon_target_and_host(target, target_host)
    remote = rpc.connect(host, port)
    n_times = 20

    bandwidth_total_item = 1 << 25
    bandwidth_item_per_thread = 32

    compute_total_item = 1 << 21
    compute_item_per_thread = 4096

    if str(target).startswith("opencl"):
        dev = remote.cl()
    elif str(target).startswith("cuda"):
        dev = remote.cuda()
    elif str(target).startswith("metal"):
        dev = remote.metal()
    else:
        raise RuntimeError("Unsupported target")

    logging.info("========== measure memory bandwidth ==========")
    measure_bandwidth_all_types(
        bandwidth_total_item, bandwidth_item_per_thread, n_times, target, target_host, remote, dev
    )

    logging.info("========== measure peak compute ==========")
    measure_compute_all_types(
        compute_total_item, compute_item_per_thread, n_times, target, target_host, remote, dev
    )
