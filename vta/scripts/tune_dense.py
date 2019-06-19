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

"""Tuning a single dense operator"""

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

Workload = namedtuple("DenseWorkload",
                      ['batch', 'in_filter', 'out_filter'])

resnet_wkls = [
    # Workloads of resnet18 on imagenet
    ('resnet-18.dense',  Workload(16, 512, 1024)),
]

@tvm.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.const(a_min, x.dtype)
    const_max = tvm.const(a_max, x.dtype)
    x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
    x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
    return x

def dense(N, CI, CO):
    data_shape = (N//env.BATCH, CI//env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO//env.BLOCK_OUT, CI//env.BLOCK_IN, env.BLOCK_OUT, env.BLOCK_IN)

    data = tvm.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = tvm.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)

    with tvm.target.vta():
        res = topi.nn.dense(data, kernel, None, 'int32')
        res = topi.right_shift(res, 8)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

    if tvm.target.current_target().device_name == 'vta':
        s = topi.generic.schedule_dense([res])
    else:
        s = tvm.create_schedule([res.op])

    return s, [data, kernel, res]

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
        CO = wl.out_filter

        task = autotvm.task.create(dense, args=(N, CI, CO),
                target=tvm.target.vta(), target_host=env.target_host, template_key='direct')
        print(task.config_space)

        measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=vta.vta_autotvm_build_func),
                runner=autotvm.RPCRunner(env.TARGET, tracket_host, tracket_port, number=4, repeat=3, timeout=10000,
                                        check_correctness=True))

        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(n_trial=len(task.config_space),
                measure_option=measure_option,
                callbacks=[autotvm.callback.log_to_file('dense.log')])

        print("\nBest tuner config:")
        print(tuner.best_config)
