import tvm
import logging
import topi
from tvm import autotvm

# task function
def conv2d():
    A = tvm.placeholder((1, 512, 7, 7), 'float32')
    B = tvm.placeholder((512, 512, 3, 3), 'float32')
    D = topi.nn.conv2d(A, B, (1,1), (1,1), 'NCHW', 'float32')
    s = topi.generic.schedule_conv2d_nchw(D)

    return s, [A, B, D]


# create task
task = autotvm.task.create(conv2d,
                           args=(),
                           target='cuda',
                           template_key='vanilla')
print(task.config_space)

# begin tuing
logging.basicConfig(level=logging.INFO)
measure_option = autotvm.measure_option(mode='local',
                                        number=10,
                                        parallel_num=8,
                                        timeout=20)
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=100,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('cache.tsv')])

# find history best
with autotvm.apply_history_best('cache.tsv'):
    with tvm.target.create("cuda"):
        s, arg_bufs = conv2d()
        func = tvm.build(s, arg_bufs)

print(tvm.lower(s, arg_bufs, simple_mode=True))

