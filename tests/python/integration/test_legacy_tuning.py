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
import textwrap

import tvm
import tvm.relay
import tvm.testing
from tvm import autotvm, te
from tvm.autotvm.measure import measure_methods
from tvm.autotvm.tuner import RandomTuner
from tvm.contrib import tar
from tvm.ir.instrument import pass_instrument
from tvm.ir.transform import PassContext
from tvm.target import Target
from tvm.tir.analysis import _ffi_api as _analysis_ffi_api


def setup_module():
    """Setup the module used for testing."""

    @autotvm.template("testing/conv2d_no_batching")
    def conv2d_no_batching(  # pylint: disable=unused-variable
        batch_size, input_h, input_w, channels_in, channels_out, kernel_h, kernel_w
    ):
        """An example template for testing"""
        assert batch_size == 1, "Only consider batch_size = 1 in this template"

        data = te.placeholder((batch_size, channels_in, input_h, input_w), name="data")
        kernel = te.placeholder((channels_out, channels_in, kernel_h, kernel_w), name="kernel")

        axis_rc = te.reduce_axis((0, channels_in), name="rc")
        axis_ry = te.reduce_axis((0, kernel_h), name="ry")
        axis_rx = te.reduce_axis((0, kernel_w), name="rx")

        conv = te.compute(
            (batch_size, channels_out, input_h - kernel_h + 1, input_w - kernel_w + 1),
            lambda nn, ff, yy, xx: te.sum(
                data[nn, axis_rc, yy + axis_ry, xx + axis_rx]
                * kernel[ff, axis_rc, axis_ry, axis_rx],
                axis=[axis_rc, axis_ry, axis_rx],
            ),
            tag="conv2d_nchw",
        )

        schedule = te.create_schedule([conv.op])

        output = conv
        cache_write_ol = schedule.cache_write(conv, "local")

        # create cache stage
        cache_read_aa = schedule.cache_read(data, "shared", [cache_write_ol])
        cache_read_ww = schedule.cache_read(kernel, "shared", [cache_write_ol])
        cache_read_al = schedule.cache_read(cache_read_aa, "local", [cache_write_ol])
        cache_read_wl = schedule.cache_read(cache_read_ww, "local", [cache_write_ol])

        # tile and bind spatial axes
        axis_n, axis_f, axis_y, axis_x = schedule[output].op.axis
        cfg = autotvm.get_config()
        cfg.define_split("tile_f", cfg.axis(axis_f), num_outputs=4)
        cfg.define_split("tile_y", cfg.axis(axis_y), num_outputs=4)
        cfg.define_split("tile_x", cfg.axis(axis_x), num_outputs=4)
        axis_bf, axis_vf, axis_tf, axis_fi = cfg["tile_f"].apply(schedule, output, axis_f)
        axis_by, axis_vy, axis_ty, axis_yi = cfg["tile_y"].apply(schedule, output, axis_y)
        axis_bx, axis_vx, axis_tx, axis_xi = cfg["tile_x"].apply(schedule, output, axis_x)
        kernel_scope = axis_n  # this is the scope to attach global config inside this kernel

        schedule[output].bind(axis_bf, te.thread_axis("blockIdx.z"))
        schedule[output].bind(axis_by, te.thread_axis("blockIdx.y"))
        schedule[output].bind(axis_bx, te.thread_axis("blockIdx.x"))
        schedule[output].bind(axis_vf, te.thread_axis("vthread"))
        schedule[output].bind(axis_vy, te.thread_axis("vthread"))
        schedule[output].bind(axis_vx, te.thread_axis("vthread"))
        schedule[output].bind(axis_tf, te.thread_axis("threadIdx.z"))
        schedule[output].bind(axis_ty, te.thread_axis("threadIdx.y"))
        schedule[output].bind(axis_tx, te.thread_axis("threadIdx.x"))
        schedule[output].reorder(
            axis_n,
            axis_bf,
            axis_by,
            axis_bx,
            axis_vf,
            axis_vy,
            axis_vx,
            axis_tf,
            axis_ty,
            axis_tx,
            axis_fi,
            axis_yi,
            axis_xi,
        )
        schedule[cache_write_ol].compute_at(schedule[output], axis_tx)

        # tile and bind reduction axes
        axis_n, axis_f, axis_y, axis_x = schedule[cache_write_ol].op.axis
        axis_rc, axis_ry, axis_rx = schedule[cache_write_ol].op.reduce_axis
        cfg.define_split("tile_rc", cfg.axis(axis_rc), num_outputs=3)
        cfg.define_split("tile_ry", cfg.axis(axis_ry), num_outputs=3)
        cfg.define_split("tile_rx", cfg.axis(axis_rx), num_outputs=3)
        axis_rco, axis_rcm, axis_rci = cfg["tile_rc"].apply(schedule, cache_write_ol, axis_rc)
        axis_ryo, axis_rym, axis_ryi = cfg["tile_rx"].apply(schedule, cache_write_ol, axis_ry)
        axis_rxo, axis_rxm, axis_rxi = cfg["tile_ry"].apply(schedule, cache_write_ol, axis_rx)
        schedule[cache_write_ol].reorder(
            axis_rco,
            axis_ryo,
            axis_rxo,
            axis_rcm,
            axis_rym,
            axis_rxm,
            axis_rci,
            axis_ryi,
            axis_rxi,
            axis_n,
            axis_f,
            axis_y,
            axis_x,
        )

        schedule[cache_read_aa].compute_at(schedule[cache_write_ol], axis_rxo)
        schedule[cache_read_ww].compute_at(schedule[cache_write_ol], axis_rxo)
        schedule[cache_read_al].compute_at(schedule[cache_write_ol], axis_rxm)
        schedule[cache_read_wl].compute_at(schedule[cache_write_ol], axis_rxm)

        # cooperative fetching
        for load in [cache_read_aa, cache_read_ww]:
            axis_n, axis_f, axis_y, axis_x = schedule[load].op.axis
            fused = schedule[load].fuse(axis_n, axis_f, axis_y, axis_x)
            axis_tz, fused = schedule[load].split(fused, nparts=cfg["tile_f"].size[2])
            axis_ty, fused = schedule[load].split(fused, nparts=cfg["tile_y"].size[2])
            axis_tx, fused = schedule[load].split(fused, nparts=cfg["tile_x"].size[2])
            schedule[load].bind(axis_tz, te.thread_axis("threadIdx.z"))
            schedule[load].bind(axis_ty, te.thread_axis("threadIdx.y"))
            schedule[load].bind(axis_tx, te.thread_axis("threadIdx.x"))

        # tune unroll
        cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
        cfg.define_knob("unroll_explicit", [0, 1])
        schedule[output].pragma(
            kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val
        )
        schedule[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

        return schedule, [data, kernel, conv]


def teardown_module():
    """Remove the module from the autotvm task tables."""
    # TODO(areusch): Tasks should not be registered into a global.
    del autotvm.task.task.TASK_TABLE["testing/conv2d_no_batching"]


def get_sample_task(target=tvm.target.cuda(), target_host=None):
    """return a sample task for testing"""
    target, target_host = Target.canon_target_and_host(target, target_host)
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
def test_tuning_gpu(target):
    """Test gpu tuning."""

    def runner(target):
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

        successful_results = [
            r
            for r in results
            if r.error_no == autotvm.MeasureErrorNo.NO_ERROR
            # We filter records before building if we know they won't work ahead of time.
            # We can't guarantee we get one good record so we count these as success too
            or r.error_no == autotvm.MeasureErrorNo.INSTANTIATION_ERROR
        ]
        assert len(successful_results) > 0, f"No successful tuning runs: {results!r}"

    run_test_with_all_multiprocessing(runner, target)


@tvm.testing.parametrize_targets("cuda", "opencl")
def test_tuning_gpu_inherits_pass_context(target):
    """Autotvm tuner inherits PassContexts but also adds a gpu verification pass by default.

    Test that using PassContext inherits passes properly but also runs gpu verification pass.
    """

    @pass_instrument
    class PassInstrumentChecker:
        """Pass Instrument that simply sees if it's been run."""

        def __init__(self):
            self.has_been_run = False

        def run_after_pass(self, *_):
            self.has_been_run = True

    class GPUVerifyPassMocked:
        """Context manager that mocks tir.analysis.verify_gpu_code meant
        to verify the pass has been run. This is done by patching the ffi func handles."""

        FFI_FUNC_HANDLE = "tir.analysis.verify_gpu_code"
        FUNC_NAME = "verify_gpu_code"

        def __init__(self) -> None:
            self.old_impl = tvm._ffi.get_global_func(self.FFI_FUNC_HANDLE)
            self.has_been_run = False

        def gpu_verify_pass_mocked(self):
            """Get the replacement for the gpu verification pass."""

            def _gpu_verify_pass_mocked(*args, **kwargs):
                self.has_been_run = True
                return self.old_impl(*args, **kwargs)

            return _gpu_verify_pass_mocked

        def __enter__(self):
            tvm._ffi.register_func(
                self.FFI_FUNC_HANDLE, self.gpu_verify_pass_mocked(), override=True
            )

            # Also overwrite the python bindings
            setattr(
                _analysis_ffi_api, self.FUNC_NAME, tvm._ffi.get_global_func(self.FFI_FUNC_HANDLE)
            )

        def __exit__(self, *args, **kwargs):
            # Restore FFI status back to normal
            tvm._ffi.register_func(self.FFI_FUNC_HANDLE, self.old_impl, override=True)
            setattr(_analysis_ffi_api, self.FUNC_NAME, self.old_impl)

    class OverwrittenBuildFunc(measure_methods._WrappedBuildFunc):
        """BuildFunc that mocks and patches as necessary to test proper passes are run."""

        def __call__(self, measure_input, tmp_dir, **kwargs):
            instrument = PassInstrumentChecker()
            mocked_pass_checker = GPUVerifyPassMocked()
            with mocked_pass_checker:
                with PassContext(instruments=[instrument]):
                    regular_result = super().__call__(measure_input, tmp_dir, **kwargs)

                    # Check instrument has been run, meaning context was inherited by builder
                    assert instrument.has_been_run

                    # But also check the gpu verification pass has been run
                    # (which was not in the inherited ctx)
                    assert mocked_pass_checker.has_been_run

                    return regular_result

    class MockedLocalBuilder(measure_methods.LocalBuilder):
        """As measure_methods.LocalBuilder but overwrites the PassContext for testing."""

        def __init__(
            self,
            timeout=10,
            n_parallel=None,
            build_kwargs=None,
            build_func="default",
            do_fork=False,
            runtime=None,
        ):
            # pylint: disable=too-many-function-args
            super().__init__(timeout, n_parallel, build_kwargs, build_func, do_fork, runtime)

            self.build_func = OverwrittenBuildFunc(tar.tar, runtime)

    def runner(target):
        task, target = get_sample_task(target, None)
        logging.info("task config space: %s", task.config_space)

        # Note: we use the MockedLocalBuilder here instead of autotvm.LocalBuilder()
        measure_option = autotvm.measure_option(MockedLocalBuilder(), autotvm.LocalRunner())

        results = []

        tuner = RandomTuner(task)
        tuner.tune(
            n_trial=1,
            measure_option=measure_option,
            callbacks=(lambda _tuner, _inputs, rs: results.extend(rs),),
        )

        assert len(results) == 1

    run_test_with_all_multiprocessing(runner, target)


def test_tuning_cpu():
    """Test tuning on cpu."""

    def runner():
        ir_mod = tvm.relay.fromtext(
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
    tvm.testing.main()
