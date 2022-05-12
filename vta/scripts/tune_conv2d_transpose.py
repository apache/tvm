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

"""Tuning a single conv2d transpose operator"""

from collections import namedtuple
import logging
import os

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import vta
import vta.testing

# Get batch info from env
env = vta.get_env()

Workload = namedtuple(
    "Conv2DTransposeWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
        "o_hpad",
        "o_wpad",
    ],
)

# DCGAN workloads
dcgan_wkls = [
    # dcgan
    ("DCGAN.CT1", Workload(env.BATCH, 4, 4, 1024, 512, 4, 4, 1, 1, 2, 2, 0, 0)),
    ("DCGAN.CT2", Workload(env.BATCH, 8, 8, 512, 256, 4, 4, 1, 1, 2, 2, 0, 0)),
    ("DCGAN.CT3", Workload(env.BATCH, 16, 16, 256, 128, 4, 4, 1, 1, 2, 2, 0, 0)),
]


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def conv2d_transpose(N, CI, H, W, CO, KH, KW, strides, padding, opadding):
    data_shape = (N // env.BATCH, CI // env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO // env.BLOCK_OUT, CI // env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)

    with tvm.target.vta():
        res = topi.nn.conv2d_transpose_nchw(
            Input=data,
            Filter=kernel,
            strides=strides,
            padding=padding,
            out_dtype=env.acc_dtype,
            output_padding=opadding,
        )
        res = topi.right_shift(res, env.WGT_WIDTH)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)

    if tvm.target.Target.current().device_name == "vta":
        s = topi.generic.schedule_conv2d_transpose_nchw([res])
    else:
        s = te.create_schedule([res.op])

    return s, [data, kernel, res]


if __name__ == "__main__":

    # Logging config (for printing tuning log to the screen)
    logging.basicConfig()
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)

    # Tuning log files
    log_file = "%s.conv2d_transpose.log" % (env.TARGET)
    # create tmp log file
    tmp_log_file = log_file + ".tmp"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Get tracker info from env
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    for idx, (wl_name, wl) in enumerate(dcgan_wkls):
        prefix = "[Task %2d/%2d] " % (idx, len(dcgan_wkls))

        # Read in workload parameters
        N = wl.batch
        H = wl.height
        W = wl.width
        CI = wl.in_filter
        CO = wl.out_filter
        KH = wl.hkernel
        KW = wl.wkernel
        strides = (wl.hstride, wl.wstride)
        padding = (wl.hpad, wl.wpad)
        opadding = (wl.o_hpad, wl.o_wpad)

        # Create task
        task = autotvm.task.create(
            conv2d_transpose,
            args=(N, CI, H, W, CO, KH, KW, strides, padding, opadding),
            target=tvm.target.Target(tvm.target.vta(), host=env.target_host),
            template_key="direct",
        )
        print(task.config_space)

        # Tune
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                env.TARGET,
                host=tracker_host,
                port=int(tracker_port),
                number=5,
                timeout=60,
                # check_correctness=True, # TODO: re-enable when check_correctness works again.
            ),
        )

        # Run Tuner
        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(
            n_trial=len(task.config_space),
            early_stopping=None,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(len(task.config_space), prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # Pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_file)
    os.remove(tmp_log_file)
