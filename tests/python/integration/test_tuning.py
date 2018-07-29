"""
Test the tuner
"""
import logging
import time

import tvm

from tvm import autotvm
from tvm.autotvm.tuner import RandomTuner

@autotvm.template
def conv2d_no_batching(N, H, W, CI, CO, KH, KW):
    """An example template for testing"""
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')

    rc = tvm.reduce_axis((0, CI), name='rc')
    ry = tvm.reduce_axis((0, KH), name='ry')
    rx = tvm.reduce_axis((0, KW), name='rx')

    conv = tvm.compute(
        (N, CO, H - KH + 1, W - KW + 1),
        lambda nn, ff, yy, xx: tvm.sum(
            data[nn, rc, yy + ry, xx + rx] * kernel[ff, rc, ry, rx],
            axis=[rc, ry, rx]), tag="conv2d_nchw")

    s = tvm.create_schedule([conv.op])

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

    return s, [data, kernel, conv]

def get_sample_task(target=tvm.target.cuda(), target_host=None):
    """return a sample task for testing"""
    task = autotvm.task.create(conv2d_no_batching,
                               args=(1, 7, 7, 512, 512, 3, 3),
                               target=target, target_host=target_host)
    return task, target


def test_task_tuner_without_measurement():
    """test task and tuner without measurement"""
    task, target = get_sample_task()

    def custom_measure(input_pack, build_func, build_args, number, repeat,
                       ref_input, ref_output):
        from tvm.autotvm import MeasureResult

        results = []
        for inp in input_pack:
            tic = time.time()
            # do nothing
            time.sleep(0.001)
            results.append(MeasureResult([time.time() - tic], 0,
                                         time.time() - tic, time.time()))
        return results
    measure_option = autotvm.measure_option(custom_measure)

    logging.info("%s", task.config_space)

    # new tuner and recorder
    for tuner_class in [autotvm.tuner.RandomTuner, autotvm.tuner.GridSearchTuner]:
        tuner = tuner_class(task)
        tuner.tune(n_trial=10, measure_option=measure_option)
        assert tuner.best_flops > 1

def test_tuning_with_measure():
    def check(target, target_host):
        ctx = tvm.context(target, 0)
        if not ctx.exist:
            logging.info("Skip test because %s is not available" % target)
            return

        # init task
        task, target = get_sample_task(target, target_host)
        logging.info("%s", task.config_space)

        measure_option = autotvm.measure_option('local',
                                                timeout=4,
                                                number=2)

        tuner = RandomTuner(task)
        tuner.tune(n_trial=10, measure_option=measure_option)

    check("cuda", None)
    check("opencl", None)

if __name__ == "__main__":
    # only print log when invoked from main
    logging.basicConfig(level=logging.DEBUG)

    test_task_tuner_without_measurement()
    test_tuning_with_measure()

