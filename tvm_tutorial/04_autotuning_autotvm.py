import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner
from tvm import relay, te
from tvm.relay.op import op as _op
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple1d
from tvm.relay.op.strategy.generic import wrap_compute_conv1d, wrap_topi_schedule

from module import TVMModule

import logging

logger = logging.getLogger("strategy")
# TVM integration: Add compute to `python/tvm/topi/nn/conv1d.py`
@autotvm.register_topi_compute("custom_conv1d_ncw")
def conv1d_ncw(cfg, data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution forward operator for NCW layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [num_filter, in_channel, filter_size]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    s = strides
    d = dilation
    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        s = strides[0]
    if isinstance(dilation, (tuple, list)):
        d = dilation[0]

    batch, in_channels, data_width = data.shape
    out_channels, _, kernel_size = kernel.shape

    # Compute padding and out width
    pad_left, pad_right = get_pad_tuple1d(padding, (kernel_size,))
    if pad_left != pad_right:
        raise ValueError("Padding has to be symmetric. Got %d %d" % pad_left, pad_right)
    p = pad_left
    out_width = (data_width + 2 * p - kernel_size - (kernel_size - 1) * (d - 1)) // s + 1

    # Compute graph
    rc = te.reduce_axis((0, in_channels), name="rc")
    rx = te.reduce_axis((0, kernel_size), name="rx")
    return te.compute(
        (batch, out_channels, out_width),
        lambda nn, kk, xx: te.sum(
            te.if_then_else(
                te.any(s * xx + d * rx - p < 0, s * xx + d * rx - p >= data_width),
                0.0,
                data[nn, rc, s * xx + d * rx - p].astype(out_dtype)
                * kernel[kk, rc, rx].astype(out_dtype),
            ),
            axis=[rc, rx],
        ),
        tag="custom_conv1d_ncw",
    )


# TVM integration: Add schedule to `python/tvm/topi/generic/nn.py`
@autotvm.register_topi_schedule("custom_conv1d_ncw")
def schedule_conv1d_ncw(cfg, outs):
    """Schedule for conv1d_ncw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv1d_ncw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    nn, kk, xx = s[outs[0]].op.axis

    cfg.define_knob("split_kk", [1, 2, 4, 8, 16])
    cfg.define_knob("split_xx", [1, 2, 4, 8, 16])

    kk_outer, kk_inner = s[outs[0]].split(kk, cfg["split_kk"].val)
    xx_outer, xx_inner = s[outs[0]].split(xx, cfg["split_xx"].val)

    s[outs[0]].reorder(kk_outer, xx_outer, kk_inner, xx_inner)
    s[outs[0]].vectorize(xx_inner)

    return s


# TVM integration: Add strategy to `python/tvm/relay/op/strategy/generic.py`
@relay.op.strategy.override_native_generic_func("custom_conv1d_strategy")
def custom_conv1d_strategy(attrs, inputs, out_type, target):
    """custom conv1d generic strategy"""
    logger.warning("custom conv1d is not optimized for this platform.")
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if layout == "NCW":
        strategy.add_implementation(
            wrap_compute_conv1d(conv1d_ncw),
            wrap_topi_schedule(schedule_conv1d_ncw),
            name="custom_conv1d_ncw.generic",
        )
    else:
        raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy


def main():
    # Load module
    m = TVMModule()
    relay_mod, relay_params = m.load()

    # Register new strategy. Default priority level is 10.
    plevel = 11
    _op.register_strategy("nn.conv1d", custom_conv1d_strategy, level=plevel)

    # Autotvm
    # logging.getLogger("autotvm").setLevel(logging.DEBUG)
    # logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    runner = autotvm.LocalRunner(
        number=10, repeat=1, timeout=10, min_repeat_ms=0, enable_cpu_cache_flush=True,
    )
    tuning_option = {
        "tuner": "xgb",
        "trials": 10,
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": "tutorial_autotvm_records.json",
    }

    target = tvm.target.Target("llvm")
    tasks = autotvm.task.extract_from_program(relay_mod["main"], target=target, params=relay_params)

    # Tune the extracted tasks sequentially.
    for i, task in enumerate(tasks):
        print(task.config_space)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

    # Compile module
    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(relay_mod, target=target, params=relay_params)

    # Benchmark module
    m.benchmark(lib)


if __name__ == "__main__":
    main()
