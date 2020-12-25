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
"""Test layout rewrite support for whole neural networks"""
import tempfile

import numpy as np

import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_runtime
import tvm.testing


def get_np_array(var, dtype):
    return np.random.randn(*[int(x) for x in var.type_annotation.shape]).astype(dtype)


def get_relay_conv2d(
    outc=128,
    inc=64,
    height=14,
    width=14,
    kh=3,
    kw=3,
    batch=1,
    pad=0,
    stride=1,
    dilation=1,
    layout="NHWC",
):
    dtype = "float32"
    if layout == "NHWC":
        kernel_layout = "HWIO"
        d = relay.var("data", shape=(batch, height, width, inc), dtype=dtype)
        w = relay.var("weight", shape=(kh, kw, inc, outc), dtype=dtype)
    elif layout == "NCHW":
        kernel_layout = "OIHW"
        d = relay.var("data", shape=(batch, inc, height, width), dtype=dtype)
        w = relay.var("weight", shape=(outc, inc, kh, kw), dtype=dtype)

    y = relay.nn.conv2d(
        d,
        w,
        padding=pad,
        kernel_size=(kh, kw),
        strides=(stride, stride),
        dilation=(dilation, dilation),
        channels=outc,
        groups=1,
        data_layout=layout,
        kernel_layout=kernel_layout,
    )
    mod = tvm.IRModule()
    mod["main"] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


def get_relay_dense(m=128, n=128, k=128):
    dtype = "float32"
    d = relay.var("data", shape=(m, k), dtype=dtype)
    w = relay.var("weight", shape=(n, k), dtype=dtype)
    y = relay.nn.dense(d, w, units=n)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


def get_relay_batchmm(batch=4, m=128, n=128, k=128):
    dtype = "float32"
    d = relay.var("data", shape=(batch, m, k), dtype=dtype)
    w = relay.var("weight", shape=(batch, n, k), dtype=dtype)
    y = relay.nn.batch_matmul(d, w)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


def tune_and_check(mod, data, weight):
    # Extract tasks from a relay program
    target = tvm.target.Target("llvm")
    tasks, task_weights = auto_scheduler.extract_tasks(mod, target=target, params={})

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        # Tune tasks
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=1,
            num_measures_per_round=1,
            builder=auto_scheduler.LocalBuilder(timeout=60),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        tuner.tune(tune_option, search_policy="sketch.random")

        # Compile and run
        def compile_and_run(disabled_pass={}):
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(
                    opt_level=3,
                    config={"relay.backend.use_auto_scheduler": True},
                    disabled_pass=disabled_pass,
                ):
                    lib = relay.build(mod, target=target, params={"weight": weight})

            ctx = tvm.cpu()
            module = graph_runtime.GraphModule(lib["default"](ctx))
            module.set_input("data", data)
            module.run()

            return module.get_output(0).asnumpy()

        # Check correctness
        actual_output = compile_and_run()
        expected_output = compile_and_run(disabled_pass={"AutoSchedulerLayoutRewrite"})

        tvm.testing.assert_allclose(actual_output, expected_output, rtol=1e-4, atol=1e-4)


def test_conv2d():
    # wrap the search in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    mod, data, weight = get_relay_conv2d(kh=1, kw=1)
    tune_and_check(mod, data, weight)


def test_dense():
    mod, data, weight = get_relay_dense()
    tune_and_check(mod, data, weight)


def test_batch_matmul():
    mod, data, weight = get_relay_batchmm()
    tune_and_check(mod, data, weight)


if __name__ == "__main__":
    test_conv2d()
    test_dense()
    test_batch_matmul()
