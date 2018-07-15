"""
How to get high performance convolution kernel on NVIDIA GPU by auto-tuning
=========================================================================
**Author**: `Lianmin Zheng <https://https://github.com/merrymercy>`_

This is an advanced tutorial for writing high performance tunable template for 
NVIDIA GPU. By running auto-tuner on this template, we can outperform the
vendor provided library CuDNN in many cases.
"""

import logging
import sys

import tvm
import topi

from tvm import autotvm

######################################################################
# Step 1:  Define the search space
# ---------------------------------
# There are plenty of useful schedule primitives in tvm. You can also find 
# some tutorials that describe them in more details, such as 
# (1). :ref:`opt-conv-gpu`
# (2). `Optimizing DepthwiseConv on NVIDIA GPU <https://tvm.ai/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example.html>`_
# 
# However, their implementations are manually tuned for some special input
# shapes. In this section, we build a large enough space to cover
# the techniques used in these tutorials. Then we rely on the efficient auto-tuner
# to search through this space and pick some good configurations.
# 
# If you are familiar with writing cuda schedule, you can find the following
# template is very general. Actually this template can be easily modified 
# to tune other operators such as depthwise convolution and gemm.
# In order to fully understand this template, you should be familiar with
# the schedule primitives and auto tuning API. You can refer to the above
# tutorials and :doc:`autotvm tutorial <tune_simple_template>`
#
# It is worth noting that the search space for a conv2d operator
# can be very large (at the level of 10^9 for some input shapes)
#

@autotvm.template
def conv2d_no_batching(N, H, W, CI, CO, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, 'float32')
    s = tvm.create_schedule([conv.op])

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, 'local')

    # create cache stage
    AA = s.cache_read(data, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])
    AL = s.cache_read(AA, 'local', [OL])
    WL = s.cache_read(WW, 'local', [OL])

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

    s[output].bind(bf, tvm.thread_axis("blockIdx.z"))
    s[output].bind(by, tvm.thread_axis("blockIdx.y"))
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile and bind reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=3)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=3)
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)
    ryo, rym, ryi = cfg['tile_rx'].apply(s, OL, ry)
    rxo, rxm, rxi = cfg['tile_ry'].apply(s, OL, rx)
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
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # tune unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

    return s, [raw_data, kernel, conv]

######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# logging config (for printing tuning log to screen)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# the last layer in resnet 
task = autotvm.task.create(conv2d_no_batching,
                           args=(1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)),
                           target='cuda')
print(task.config_space)

# use local gpu, measure 5 times for every config to reduce variance
# run 8 parallel threads for compilation
measure_option = autotvm.measure_option(mode='local',
                                        number=10,
                                        parallel_num=8,
                                        timeout=20)

# begin tuning, log records to file `cache.tsv`
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=20,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('cache.tsv')])

# get best config from cache file
dispatch_context = autotvm.apply_history_best("cache.tsv")
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

