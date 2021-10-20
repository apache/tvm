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
from tvm import te
from tvm import autotvm
from tvm import topi
import vta
import vta.testing

env = vta.get_env()

Workload = namedtuple("DenseWorkload", ["batch", "in_filter", "out_filter"])

dense_wkls = [
    ("lstm.dense.1", Workload(1, 256, 128)),
    ("lstm.dense.4", Workload(4, 256, 128)),
]


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def dense(N, CI, CO):
    data_shape = (N // env.BATCH, CI // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO // env.BLOCK_OUT, CI // env.BLOCK_IN, env.BLOCK_OUT, env.BLOCK_IN)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)

    with tvm.target.vta():
        res = topi.nn.dense(data, kernel, None, "int32")
        res = topi.right_shift(res, 8)
        res = my_clip(res, 0, 127)
        res = topi.cast(res, "int8")

    if tvm.target.Target.current().device_name == "vta":
        s = topi.generic.schedule_dense([res])
    else:
        s = te.create_schedule([res.op])

    return s, [data, kernel, res]


if __name__ == "__main__":

    # Logging config (for printing tuning log to the screen)
    logging.basicConfig()
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)

    # Tuning log files
    log_file = "%s.dense.log" % (env.TARGET)
    # create tmp log file
    tmp_log_file = log_file + ".tmp"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Get tracker info from env
    tracket_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracket_port = os.environ.get("TVM_TRACKER_PORT", None)
    if not tracket_host or not tracket_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    for idx, (wl_name, wl) in enumerate(dense_wkls):

        prefix = "[Task %2d/%2d] " % (idx, len(dense_wkls))

        # Workload parameters
        N = wl.batch
        CI = wl.in_filter
        CO = wl.out_filter

        task = autotvm.task.create(
            dense,
            args=(N, CI, CO),
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
                host=tracket_host,
                port=int(tracket_port),
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
