import logging
import sys

import numpy as np
import tvm

from tvm import autotvm

@autotvm.template
def matmul(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


N, L, M = 512, 512, 512
task = autotvm.task.create(matmul, args=(N, L, M, 'float32'), target='c')
print(task.config_space)

# logging config (for printing tuning log to the screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

TOOLCHAIN_PREFIX = 'arm-none-eabi-'
import tvm.micro as micro

def test_build_func(obj_path, src_paths, **kwargs):
    assert len(src_paths) == 1
    cross_compiler = tvm.micro.cross_compiler(TOOLCHAIN_PREFIX, micro.LibType.OPERATOR)
    cross_compiler(obj_path, src_paths)
    input('check obj')
test_build_func.output_format = 'obj'

# TODO(weberlo): look at `tune_relay_vta.py` and `vta.exec.rpc_server` to
# figure out how to override the default rpc server with utvm infra. merge in
# what you've done in `test_micro_rpc.py`.

# There are two steps for measuring a config: build and run.
# By default, we use all CPU cores to compile program. Then measure them sequentially.
# We measure 5 times and take average to reduce variance.

#measure_option = autotvm.measure_option(
#    builder=autotvm.LocalBuilder(
#        build_func=tvm.micro.cross_compiler(TOOLCHAIN_PREFIX, micro.LibType.OPERATOR), n_parallel=1, do_fork=False),
#    runner=autotvm.LocalRunner(number=5))
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(
        build_func=test_build_func, n_parallel=1, do_fork=False),
    # TODO(webelrl)o: we need to make the local runner use utvm infra
    runner=autotvm.LocalRunner(number=5))

# begin tuning, log records to file `matmul.log`
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=10,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])

# apply history best from log file
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create("llvm"):
        s, arg_bufs = matmul(N, L, M, 'float32')
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
