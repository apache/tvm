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

"""Tuning a single conv2d operator"""

from collections import namedtuple
import logging
import os

import tvm
from tvm import autotvm
from tvm.contrib.util import get_lower_ir
import topi
import vta
import vta.testing

env = vta.get_env()

Workload = namedtuple("Conv2DWorkload",
                      ['batch', 'height', 'width', 'in_filter', 'out_filter',
                       'hkernel', 'wkernel', 'hpad', 'wpad', 'hstride', 'wstride'])

resnet_wkls = [
    # Workloads of resnet18 on imagenet
    # ('resnet-18.C1',  Workload(1, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
    ('resnet-18.C2',  Workload(1,  56,  56, 64,  64,  3, 3, 1, 1, 1, 1)),
    # ('resnet-18.C3',  Workload(1,  56,  56, 64,  64,  1, 1, 0, 0, 1, 1)), # this layer does not appear in ResNet
    ('resnet-18.C4',  Workload(1,  56,  56, 64,  128, 3, 3, 1, 1, 2, 2)),
    ('resnet-18.C5',  Workload(1,  56,  56, 64,  128, 1, 1, 0, 0, 2, 2)),
    ('resnet-18.C6',  Workload(1,  28,  28, 128, 128, 3, 3, 1, 1, 1, 1)),
    ('resnet-18.C7',  Workload(1,  28,  28, 128, 256, 3, 3, 1, 1, 2, 2)),
    ('resnet-18.C8',  Workload(1,  28,  28, 128, 256, 1, 1, 0, 0, 2, 2)),
    ('resnet-18.C9',  Workload(1,  14,  14, 256, 256, 3, 3, 1, 1, 1, 1)),
    ('resnet-18.C10', Workload(1,  14,  14, 256, 512, 3, 3, 1, 1, 2, 2)),
    ('resnet-18.C11', Workload(1,  14,  14, 256, 512, 1, 1, 0, 0, 2, 2)),
    ('resnet-18.C12', Workload(1,   7,   7, 512, 512, 3, 3, 1, 1, 1, 1)),
]

@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

def conv2d(N, CI, H, W, CO, KH, KW, strides, padding, dilation, in_dtype, out_dtype):
    data_shape = (N//env.BATCH, CI//env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO//env.BLOCK_OUT, CI//env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)
    bias_shape = (N//env.BATCH, CO//env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)

    data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    bias = tvm.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)
    kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)

    with tvm.target.vta():
        res = topi.nn.conv2d(data, kernel, padding=padding, strides=strides, dilation=dilation,
                             layout='NCHW%dn%dc' % (env.BATCH, env.BLOCK_IN), out_dtype='int32')
        res = topi.add(res, bias)
        res = topi.right_shift(res, 8)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

    if tvm.target.current_target().device_name == 'vta':
        s = topi.generic.schedule_conv2d_nchw([res])
    else:
        s = tvm.create_schedule([res.op])

    return s, [data, kernel, bias, res]

if __name__ == '__main__':

    # Logging config (for printing tuning log to the screen)
    logging.basicConfig()
    logging.getLogger('autotvm').setLevel(logging.DEBUG)

    # Get tracker info from env
    tracket_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracket_port = int(os.environ.get("TVM_TRACKER_PORT", None))
    if not tracket_host or not tracket_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    for wl_name, wl in resnet_wkls:

        # Workload parameters
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
        in_dtype = 'int8'
        out_dtype = 'int32'

        task = autotvm.task.create(conv2d, args=(N, CI, H, W, CO, KH, KW, strides, padding, dilation, in_dtype, out_dtype),
                target=tvm.target.vta(), target_host=env.target_host, template_key='direct')
        print(task.config_space)

        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=vta.vta_autotvm_build_func),
                runner=autotvm.RPCRunner(env.TARGET, tracket_host, tracket_port, number=4, repeat=3, timeout=10000,
                                        check_correctness=True))

        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=len(task.config_space),
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file('conv2d.log')])

        print("\nBest tuner config:")
        print(tuner.best_config)
