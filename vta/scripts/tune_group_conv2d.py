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

"""Tuning a single group conv2d operator"""

from collections import namedtuple
import logging
import os

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import vta
import vta.testing

env = vta.get_env()

Workload = namedtuple(
    "GroupConv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "groups",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

# Mobilenet (grouped variant) workloads
mobilenet_wkls = [
    ("mobilenet.D1", Workload(env.BATCH, 112, 112, 32, 32, 2, 3, 3, 1, 1, 1, 1)),
    ("mobilenet.D2", Workload(env.BATCH, 112, 112, 64, 64, 4, 3, 3, 1, 1, 2, 2)),
    ("mobilenet.D3", Workload(env.BATCH, 56, 56, 128, 128, 8, 3, 3, 1, 1, 1, 1)),
    ("mobilenet.D4", Workload(env.BATCH, 56, 56, 128, 128, 8, 3, 3, 1, 1, 2, 2)),
    ("mobilenet.D5", Workload(env.BATCH, 28, 28, 256, 256, 16, 3, 3, 1, 1, 1, 1)),
    ("mobilenet.D6", Workload(env.BATCH, 28, 28, 256, 256, 16, 3, 3, 1, 1, 2, 2)),
    ("mobilenet.D7", Workload(env.BATCH, 14, 14, 512, 512, 32, 3, 3, 1, 1, 1, 1)),
    ("mobilenet.D8", Workload(env.BATCH, 14, 14, 512, 512, 32, 3, 3, 1, 1, 2, 2)),
    ("mobilenet.D9", Workload(env.BATCH, 7, 7, 1024, 1024, 64, 3, 3, 1, 1, 1, 1)),
]


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def group_conv2d(N, CI, H, W, CO, KH, KW, strides, padding, dilation, group):

    CI_G = CI // groups
    data_shape = (N // env.BATCH, CI // env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO // env.BLOCK_OUT, CI_G // env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)
    bias_shape = (N // env.BATCH, CO // env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)

    with tvm.target.vta():
        res = topi.nn.group_conv2d_nchw(
            data, kernel, strides, padding, dilation, groups, env.acc_dtype
        )
        res = topi.right_shift(res, env.WGT_WIDTH)
        res = topi.add(res, bias)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)

    if tvm.target.Target.current().device_name == "vta":
        s = topi.generic.schedule_group_conv2d_nchw([res])
    else:
        s = te.create_schedule([res.op])

    return s, [data, kernel, bias, res]


if __name__ == "__main__":

    # Logging config (for printing tuning log to the screen)
    logging.basicConfig()

    # Tuning log files
    log_file = "%s.group_conv2d.log" % (env.TARGET)
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

    for idx, (wl_name, wl) in enumerate(mobilenet_wkls):
        prefix = "[Task %2d/%2d] " % (idx, len(mobilenet_wkls))

        # Read in workload parameters
        N = wl.batch
        CI = wl.in_filter
        H = wl.height
        W = wl.width
        CO = wl.out_filter
        KH = wl.hkernel
        KW = wl.wkernel
        strides = (wl.hstride, wl.wstride)
        padding = (wl.hpad, wl.wpad)
        dilation = (1, 1)
        groups = wl.groups

        # Create task
        task = autotvm.task.create(
            group_conv2d,
            args=(N, CI, H, W, CO, KH, KW, strides, padding, dilation, groups),
            target=tvm.target.vta(),
            target_host=env.target_host,
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
