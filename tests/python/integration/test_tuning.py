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
Test the tuner
"""
import logging
import multiprocessing as mp
import sys
import textwrap
import time

import pytest
import tvm
import tvm.relay
import tvm.testing
from tvm import autotvm, te
from tvm.autotvm.tuner import RandomTuner
from tvm.target import Target


def setup_module():
    @autotvm.template("testing/conv2d_no_batching")
    def conv2d_no_batching(N, H, W, CI, CO, KH, KW):
        """An example template for testing"""
        assert N == 1, "Only consider batch_size = 1 in this template"

        data = te.placeholder((N, CI, H, W), name="data")
        kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

        rc = te.reduce_axis((0, CI), name="rc")
        ry = te.reduce_axis((0, KH), name="ry")
        rx = te.reduce_axis((0, KW), name="rx")

        conv = te.compute(
            (N, CO, H - KH + 1, W - KW + 1),
            lambda nn, ff, yy, xx: te.sum(
                data[nn, rc, yy + ry, xx + rx] * kernel[ff, rc, ry, rx], axis=[rc, ry, rx]
            ),
            tag="conv2d_nchw",
        )

        s = te.create_schedule([conv.op])

        output = conv
        OL = s.cache_write(conv, "local")

        # create cache stage
        AA = s.cache_read(data, "shared", [OL])
        WW = s.cache_read(kernel, "shared", [OL])
        AL = s.cache_read(AA, "local", [OL])
        WL = s.cache_read(WW, "local", [OL])

        # tile and bind spatial axes
        n, f, y, x = s[output].op.axis
        cfg = autotvm.get_config()
        cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
        cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
        cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
        bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
        by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
        bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
        kernel_scope = n  # this is the scope to attach global config inside this kernel

        s[output].bind(bf, te.thread_axis("blockIdx.z"))
        s[output].bind(by, te.thread_axis("blockIdx.y"))
        s[output].bind(bx, te.thread_axis("blockIdx.x"))
        s[output].bind(vf, te.thread_axis("vthread"))
        s[output].bind(vy, te.thread_axis("vthread"))
        s[output].bind(vx, te.thread_axis("vthread"))
        s[output].bind(tf, te.thread_axis("threadIdx.z"))
        s[output].bind(ty, te.thread_axis("threadIdx.y"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))
        s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
        s[OL].compute_at(s[output], tx)

        # tile and bind reduction axes
        n, f, y, x = s[OL].op.axis
        rc, ry, rx = s[OL].op.reduce_axis
        cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
        cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=3)
        cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=3)
        rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
        ryo, rym, ryi = cfg["tile_rx"].apply(s, OL, ry)
        rxo, rxm, rxi = cfg["tile_ry"].apply(s, OL, rx)
        s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

        s[AA].compute_at(s[OL], rxo)
        s[WW].compute_at(s[OL], rxo)
        s[AL].compute_at(s[OL], rxm)
        s[WL].compute_at(s[OL], rxm)

        # cooperative fetching
        for load in [AA, WW]:
            n, f, y, x = s[load].op.axis
            fused = s[load].fuse(n, f, y, x)
            tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
            ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
            tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
            s[load].bind(tz, te.thread_axis("threadIdx.z"))
            s[load].bind(ty, te.thread_axis("threadIdx.y"))
            s[load].bind(tx, te.thread_axis("threadIdx.x"))

        # tune unroll
        cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
        cfg.define_knob("unroll_explicit", [0, 1])
        s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

        return s, [data, kernel, conv]


def teardown_module():
    # TODO(areusch): Tasks should not be registered into a global.
    del autotvm.task.task.TASK_TABLE["testing/conv2d_no_batching"]


def get_sample_task(target=tvm.target.cuda(), target_host=None):
    target, target_host = Target.check_and_update_host_consist(target, target_host)
    """return a sample task for testing"""
    task = autotvm.task.create(
        "testing/conv2d_no_batching", args=(1, 7, 7, 512, 512, 3, 3), target=target
    )
    return task, target


def run_test_with_all_multiprocessing(func, *args, **kwargs):
    """Check all multiprocessing methods work for the tuning test.

    In the past fork() had the most support at detriment to spawn() and forkserver().
    As fork() is unavailable or unsafe on some platforms it is good to check all
    available methods.
    """
    for multiprocessing_method in mp.get_all_start_methods():
        old_start_method = mp.get_start_method()
        try:
            mp.set_start_method(multiprocessing_method, force=True)
            func(*args, **kwargs)
        finally:
            mp.set_start_method(old_start_method, force=True)


@tvm.testing.parametrize_targets("cuda", "opencl")
def test_tuning_gpu(target, dev):
    def runner(target, dev):
        # init task
        task, target = get_sample_task(target, None)
        logging.info("task config space: %s", task.config_space)

        measure_option = autotvm.measure_option(autotvm.LocalBuilder(), autotvm.LocalRunner())

        results = []

        tuner = RandomTuner(task)
        tuner.tune(
            n_trial=20,
            measure_option=measure_option,
            callbacks=(lambda _tuner, _inputs, rs: results.extend(rs),),
        )

        assert len(results) == 20

        successful_results = [r for r in results if r.error_no == autotvm.MeasureErrorNo.NO_ERROR]
        assert len(successful_results) > 0, f"No successful tuning runs: {results!r}"

    run_test_with_all_multiprocessing(runner, target, dev)


def test_tuning_cpu():
    def runner():
        ir_mod = tvm.parser.fromtext(
            textwrap.dedent(
                """
            #[version = "0.0.5"]
            def @main(%a : Tensor[(1, 3, 32, 32), float32], %b : Tensor[(3, 3, 5, 5), float32]) {
                nn.conv2d(%a, %b, data_layout="NCHW", kernel_layout="OIHW")
            }
            """
            )
        )
        tasks = autotvm.task.relay_integration.extract_from_program(
            ir_mod, {}, tvm.target.create("llvm")
        )
        assert len(tasks) == 1, f"Extracted != 1 task from program: {tasks!r}"

        task = tasks[0]

        measure_option = autotvm.measure_option(autotvm.LocalBuilder(), autotvm.LocalRunner())

        results = []

        tuner = RandomTuner(task)
        tuner.tune(
            n_trial=20,
            measure_option=measure_option,
            callbacks=(lambda _tuner, _inputs, rs: results.extend(rs),),
        )

        assert len(results) == 20

        successful_results = [r for r in results if r.error_no == autotvm.MeasureErrorNo.NO_ERROR]
        assert len(successful_results) > 0, f"No successful tuning runs: {results!r}"

    run_test_with_all_multiprocessing(runner)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
