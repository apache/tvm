
from __future__ import absolute_import as _abs
from os import wait
from tvm import te, tir, relay, autotvm, topi
import re
from tvm.topi.utils import get_const_tuple
from ..nn import conv2d_legalize, conv1d_legalize
import tvm


def sdotp(data_dtype, kernel_dtype, out_dtype, vec_length, data_is_last_axis=True, kernel_is_last_axis=True, dilation=1, axis_reordered=False):

    flip = False
    if data_dtype.startswith("u"):
        if kernel_dtype.startswith("u"):
            intrinsic_name = "llvm.riscv.pulp.sdotup"
        else:
            intrinsic_name = "llvm.riscv.pulp.sdotusp"
    else:
        if kernel_dtype.startswith("u"):
            intrinsic_name = "llvm.riscv.pulp.sdotusp"
            flip = True
        else:
            intrinsic_name = "llvm.riscv.pulp.sdotsp"

    intrinsic_name += str(vec_length)

    a_size = (vec_length - 1) * dilation + 1
    a_shape = (a_size,) if data_is_last_axis else (a_size, 1)
    a = te.placeholder(a_shape, data_dtype, "a")

    b_shape = (vec_length,) if kernel_is_last_axis else (vec_length, 1)
    b = te.placeholder(b_shape, kernel_dtype, "b")

    k = te.reduce_axis((0, vec_length), "k")
    l = te.reduce_axis((0, 1), "l")

    ak = a[k * dilation] if data_is_last_axis else a[k * dilation, 0]
    ak = ak.astype(out_dtype)

    bk = b[k] if kernel_is_last_axis else b[k, 0]
    bk = bk.astype(out_dtype)

    reduce_axis = [k, l] if axis_reordered else [k]
    c = te.compute((1,), lambda _: te.sum(ak * bk, reduce_axis), "c")

    def intrin_func(ins, outs):
        aa, bb = ins
        cc,    = outs

        #no easier way to build a vector?
        if data_is_last_axis and dilation == 1:
            aaval = aa.vload([0], data_dtype + "x" + str(vec_length))
        else:
            aaval = tir.Load(data_dtype + "x" + str(vec_length), aa.data,
                             tir.Ramp(aa.elem_offset, dilation * aa.strides[0], vec_length))

        if kernel_is_last_axis:
            bbval = bb.vload([0], kernel_dtype + "x" + str(vec_length))
        else:
            bbval = tir.Load(kernel_dtype + "x" + str(vec_length), bb.data,
                             tir.Ramp(bb.elem_offset, bb.strides[0], vec_length))

        if flip:
            aaval, bbval = bbval, aaval

        ccval = cc.vload([0]).astype("int32")

        call = tir.call_llvm_pure_intrin(
            "int32",
            intrinsic_name,
            3,
            aaval, bbval, ccval
        ).astype(out_dtype)

        body = cc.vstore([0], call)

        reduce_init = cc.vstore([0], tir.IntImm(out_dtype, 0))
        return None, reduce_init, body

    Ab = tir.decl_buffer(a.shape, a.dtype, name="A",
                         offset_factor=1, strides=[1] if data_is_last_axis else [te.var("batch"), 1], data_alignment=1)
    Bb = tir.decl_buffer(b.shape, b.dtype, name="B",
                         offset_factor=1, strides=[1] if kernel_is_last_axis else [te.var("out_channels"), 1], data_alignment=1)
    Cb = tir.decl_buffer(c.shape, c.dtype, name="C",
                         offset_factor=1, strides=[1], data_alignment=1)

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


@autotvm.register_topi_schedule("conv1d_ncw.pulp")
def schedule_conv1d_ncw(cfg, outs):
    out = outs[0]
    data, weight = out.op.input_tensors
    s = te.create_schedule([out.op])
    n, c, w = out.op.axis
    rc, rw = out.op.reduce_axis

    workload = out.op.attrs["workload"]
    dilation = get_const_tuple(workload[5])

    target = tvm.target.target.Target.current()

    if re.match("u?int(8|16|32)", out.dtype) and target.kind.name == "llvm":

        if re.match("u?int8", data.dtype) and re.match("u?int8", weight.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", weight.dtype):
            vec_length = 2
        else:
            vec_length = 0

        if vec_length:
            rwo, rwi = s[out].split(rw, vec_length)
            t = sdotp(data.dtype, weight.dtype, out.dtype,
                      vec_length, dilation=dilation[0])
            s[out].tensorize(rwi, t)

            cfg.define_split("tile_w", w, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            wo, wi = cfg["tile_w"].apply(s, out, w)
            cfg.define_split("tile_rwo", rwo, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            rwoo, rwoi = cfg["tile_rwo"].apply(s, out, rwo)

            s[out].reorder(n, wo, rwoo, c, rc, wi, rwoi, rwi)

    return s

@autotvm.register_topi_compute("conv1d_ncw.pulp")
def conv1d_ncw(cfg, data, kernel, strides=(1,), padding="VALID", dilation=(1,), out_dtype=None):

    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    if re.match("u?int(8|16|32)", out_dtype):
        if re.match("u?int8", data.dtype) and re.match("u?int8", kernel.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", kernel.dtype):
            vec_length = 2
        else:
            vec_length = 0

    if vec_length:
        rem = kernel.shape[2] % vec_length
        if rem:
            data = topi.nn.pad(data, (0, 0, 0), (0, 0, vec_length - rem))
            kernel = topi.nn.pad(kernel, (0, 0, 0), (0, 0, vec_length - rem))

    conv = topi.nn.conv1d_ncw(data, kernel, strides, padding, dilation, out_dtype)

    N, _, IW = data.shape
    OC, IC, KW = kernel.shape
    _, _, OW = conv.shape
    cfg.add_flop(2 * N * OW * OC * IC * ((KW - 1) * dilation + 1))
    return conv


@autotvm.register_topi_schedule("conv1d_nwc.pulp")
def schedule_conv1d_nwc(cfg, outs):
    out = outs[0]
    data, weight = out.op.input_tensors
    s = te.create_schedule([out.op])
    n, w, c = out.op.axis
    rc, rw = out.op.reduce_axis

    target = tvm.target.target.Target.current()

    if re.match("u?int(8|16|32)", out.dtype) and target.kind.name == "llvm":

        if re.match("u?int8", data.dtype) and re.match("u?int8", weight.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", weight.dtype):
            vec_length = 2
        else:
            vec_length = 0

        if vec_length:
            t = sdotp(data.dtype, weight.dtype, out.dtype,
                      vec_length,
                      kernel_is_last_axis=False, axis_reordered=True)
            s[out].reorder(rw, rc)
            rco, rci = s[out].split(rc, vec_length)
            s[out].tensorize(rci, t)

            cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            co, ci = cfg["tile_c"].apply(s, out, c)
            cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

            s[out].reorder(n, co, rcoo, w, rw, ci, rcoi)

    return s


@autotvm.register_topi_compute("conv1d_nwc.pulp")
def conv1d_nwc(cfg, data, kernel, strides=(1,), padding="VALID", dilation=(1,), out_dtype=None):

    if re.match("u?int(8|16|32)", out_dtype):
        if re.match("u?int8", data.dtype) and re.match("u?int8", kernel.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", kernel.dtype):
            vec_length = 2
        else:
            vec_length = 0

    if vec_length:
        rem = data.shape[2] % vec_length
        if rem:
            data = topi.nn.pad(data, (0, 0, 0), (0, 0, vec_length - rem))
            kernel = topi.nn.pad(kernel, (0, 0, 0), (0, vec_length - rem, 0))

    return topi.nn.conv1d_nwc(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv1d_nwc_owi.pulp")
def schedule_conv1d_nwc_owi(cfg, outs):
    out = outs[0]
    data, weight = out.op.input_tensors
    s = te.create_schedule([out.op])
    n, w, c = out.op.axis
    rw, rc = out.op.reduce_axis

    target = tvm.target.target.Target.current()

    if re.match("u?int(8|16|32)", out.dtype) and target.kind.name == "llvm":

        if re.match("u?int8", data.dtype) and re.match("u?int8", weight.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", weight.dtype):
            vec_length = 2
        else:
            vec_length = 0

        if vec_length:
            t = sdotp(data.dtype, weight.dtype, out.dtype,
                      vec_length)
            rco, rci = s[out].split(rc, vec_length)
            s[out].tensorize(rci, t)

            cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            co, ci = cfg["tile_c"].apply(s, out, c)
            cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

            s[out].reorder(n, rcoo, w, rw, co, ci, rcoi, rci)

    return s


@autotvm.register_topi_compute("conv1d_nwc_owi.pulp")
def conv1d_nwc_owi(cfg, data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):

    if re.match("u?int(8|16|32)", out_dtype):
        if re.match("u?int8", data.dtype) and re.match("u?int8", kernel.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", kernel.dtype):
            vec_length = 2
        else:
            vec_length = 0

    if vec_length:
        rem = data.shape[2] % vec_length
        if rem:
            data = topi.nn.pad(data, (0, 0, 0), (0, 0, vec_length - rem))
            kernel = topi.nn.pad(kernel, (0, 0, 0), (0, 0, vec_length - rem))

    return topi.nn.conv1d_nwc_owi(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nchw.pulp")
def schedule_conv2d_nchw(cfg, outs):
    out = outs[0]
    data, weight = out.op.input_tensors

    s = te.create_schedule([out.op])
    n, c, h, w = out.op.axis
    rc, ry, rx = out.op.reduce_axis

    workload = out.op.attrs["workload"]
    dilation = get_const_tuple(workload[5])

    target = tvm.target.target.Target.current()

    if re.match("u?int(8|16|32)", out.dtype) and target.kind.name == "llvm":

        if re.match("u?int8", data.dtype) and re.match("u?int8", weight.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", weight.dtype):
            vec_length = 2
        else:
            vec_length = 0

        if vec_length and weight.shape[2] % vec_length == 0:
            t = sdotp(data.dtype, weight.dtype, out.dtype,
                      vec_length, dilation=dilation[1])
            rxo, rxi = s[out].split(rx, vec_length)
            s[out].tensorize(rxi, t)

            cfg.define_split("tile_w", w, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            wo, wi = cfg["tile_w"].apply(s, out, w)
            cfg.define_split("tile_rxo", rxo, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            rxoo, rxoi = cfg["tile_rxo"].apply(s, out, rxo)

            s[out].reorder(n, wo, rxoo, h, c, rc, ry, wi, rxoi)

    return s

@autotvm.register_topi_compute("conv2d_nchw.pulp")
def conv2d_nchw(cfg, data, kernel, stride, padding, dilation, out_dtype=None):

    if re.match("u?int(8|16|32)", out_dtype):
        if re.match("u?int8", data.dtype) and re.match("u?int8", kernel.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", kernel.dtype):
            vec_length = 2
        else:
            vec_length = 0

    if vec_length:
        rem = kernel.shape[3] % vec_length
        if rem:
            data = topi.nn.pad(data, (0, 0, 0, 0), (0, 0, 0, vec_length - rem))
            kernel = topi.nn.pad(kernel, (0, 0, 0, 0), (0, 0, 0, vec_length - rem))

    return topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc.pulp")
def schedule_conv2d_nhwc(cfg, outs):
    out = outs[0]
    data, weight = out.op.input_tensors
    s = te.create_schedule([out.op])
    n, h, w, c = out.op.axis
    ry, rx, rc = out.op.reduce_axis

    target = tvm.target.target.Target.current()

    if re.match("u?int(8|16|32)", out.dtype) and target.kind.name == "llvm":

        if re.match("u?int8", data.dtype) and re.match("u?int8", weight.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", weight.dtype):
            vec_length = 2
        else:
            vec_length = 0

        if vec_length:
            t = sdotp(data.dtype, weight.dtype, out.dtype,
                      vec_length, kernel_is_last_axis=False)
            rco, rci = s[out].split(rc, vec_length)
            s[out].tensorize(rci, t)

            cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            co, ci = cfg["tile_c"].apply(s, out, c)
            cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

            s[out].reorder(n, co, rcoo, h, w, ry, rx, ci, rcoi)

    return s

@autotvm.register_topi_compute("conv2d_nhwc.pulp")
def conv2d_nhwc(cfg, data, kernel, stride, padding, dilation, out_dtype=None):

    if re.match("u?int(8|16|32)", out_dtype):
        if re.match("u?int8", data.dtype) and re.match("u?int8", kernel.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", kernel.dtype):
            vec_length = 2
        else:
            vec_length = 0

    if vec_length:
        rem = data.shape[3] % vec_length
        if rem:
            data = topi.nn.pad(data, (0, 0, 0, 0), (0, 0, 0, vec_length - rem))
            kernel = topi.nn.pad(kernel, (0, 0, 0, 0), (0, 0, vec_length - rem, 0))

    return topi.nn.conv2d_nhwc(data, kernel, stride, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_nhwc_ohwi.pulp")
def schedule_conv2d_nhwc_ohwi(cfg, outs):
    out = outs[0]
    data, weight = out.op.input_tensors
    s = te.create_schedule([out.op])
    n, h, w, c = out.op.axis
    ry, rx, rc = out.op.reduce_axis

    target = tvm.target.target.Target.current()

    if re.match("u?int(8|16|32)", out.dtype) and target.kind.name == "llvm":

        if re.match("u?int8", data.dtype) and re.match("u?int8", weight.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", weight.dtype):
            vec_length = 2
        else:
            vec_length = 0

        if vec_length:
            t = sdotp(data.dtype, weight.dtype, out.dtype,
                      vec_length)
            rco, rci = s[out].split(rc, vec_length)
            s[out].tensorize(rci, t)

            cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            co, ci = cfg["tile_c"].apply(s, out, c)
            cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
            rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

            s[out].reorder(n, co, rcoo, h, w, ry, rx, ci, rcoi)

    return s

@autotvm.register_topi_compute("conv2d_nhwc_ohwi.pulp")
def conv2d_nhwc_ohwi(cfg, data, kernel, stride, padding, dilation, out_dtype=None):

    if re.match("u?int(8|16|32)", out_dtype):
        if re.match("u?int8", data.dtype) and re.match("u?int8", kernel.dtype):
            vec_length = 4
        elif re.match("u?int16", data.dtype) and re.match("u?int16", kernel.dtype):
            vec_length = 2
        else:
            vec_length = 0

    if vec_length:
        rem = data.shape[3] % vec_length
        if rem:
            data = topi.nn.pad(data, (0, 0, 0, 0), (0, 0, 0, vec_length - rem))
            kernel = topi.nn.pad(kernel, (0, 0, 0, 0), (0, 0, 0, vec_length - rem))

    return topi.nn.conv2d_nhwc_ohwi(data, kernel, stride, padding, dilation, out_dtype)


# @conv1d_legalize.register("pulp")
# def _conv1d_legalize(attrs, inputs, arg_types):
#     data, weight = inputs
#     data_type, weight_type, out_type = arg_types
#     data_layout = attrs.data_layout
#     kernel_layout = attrs.kernel_layout

#     if re.match("u?int(8|16|32)", out_type.dtype):
#         if re.match("u?int8", data_type.dtype) and re.match("u?int8", weight_type.dtype):
#             vec_length = 4
#         elif re.match("u?int16", data_type.dtype) and re.match("u?int16", weight_type.dtype):
#             vec_length = 2
#         else:
#             vec_length = 0
#     else:
#         return None

#     if data_layout == "NCW" and kernel_layout == "OIW" and vec_length:
#         out_channels, in_channels, width = weight_type.shape
#         rem = width % vec_length
#         if rem:
#             weight = relay.nn.pad(weight, ((0,0), (0,0), (0, vec_length - rem)))
#             data = relay.nn.pad(data, ((0,0), (0,0), (0, vec_length - rem)))
#             new_attrs = {k: attrs[k] for k in attrs.keys()}
#             new_attrs["kernel_size"] = (width + vec_length - rem,)
#             return relay.nn.conv1d(data, weight, **new_attrs)

#     elif data_layout == "NWC" and vec_length:
#         batch, width, in_channels = data_type.shape
#         rem = in_channels % vec_length
#         if rem:
#             if kernel_layout == "WIO":
#                 weight = relay.nn.pad(weight, ((0,0), (0, vec_length - rem), (0,0)))
#             elif kernel_layout == "OWI":
#                 weight = relay.nn.pad(weight, ((0,0), (0,0), (0, vec_length - rem)))
#             else:
#                 return None

#             data = relay.nn.pad(data, ((0,0), (0,0), (0, vec_length - rem)))
#             return relay.nn.conv1d(data, weight, **attrs)
#     return None


# @conv2d_legalize.register("pulp")
# def _conv2d_legalize(attrs, inputs, arg_types):
#     data, weight = inputs
#     data_type, weight_type, out_type = arg_types
#     data_layout = attrs.data_layout
#     kernel_layout = attrs.kernel_layout

#     if re.match("u?int(8|16|32)", out_type.dtype):
#         if re.match("u?int8", data_type.dtype) and re.match("u?int8", weight_type.dtype):
#             vec_length = 4
#         elif re.match("u?int16", data_type.dtype) and re.match("u?int16", weight_type.dtype):
#             vec_length = 2
#         else:
#             vec_length = 0
#     else:
#         return None

#     if data_layout == "NCHW" and vec_length:
#         out_channels, in_channels, height, width = weight_type.shape
#         rem = width % vec_length
#         if rem:
#             data = relay.nn.pad(data, ((0,0), (0,0), (0,0), (0, vec_length - rem)))
#             weight = relay.nn.pad(weight, ((0,0), (0,0), (0,0), (0, vec_length - rem)))
#             new_attrs = {k: attrs[k] for k in attrs.keys()}
#             new_attrs["kernel_size"] = (height, width + vec_length - rem)
#             return relay.nn.conv2d(data, weight, **new_attrs)

#     elif data_layout == "NHWC" and vec_length:
#         batch, height, width, in_channels = data_type.shape
#         kernel_height, kernel_width, _, out_channels = weight_type.shape
#         rem = in_channels % vec_length
#         if rem:
#             data = relay.nn.pad(data, ((0,0), (0,0), (0,0), (0, vec_length - rem)))
#             if kernel_layout == "HWIO":
#                 weight = relay.nn.pad(weight, ((0,0), (0,0), (0, vec_length - rem), (0,0)))
#             elif kernel_layout == "OHWI":
#                 weight = relay.nn.pad(weight, ((0,0), (0,0), (0,0), (0, vec_length - rem)))
#             else:
#                 return None
#             return relay.nn.conv2d(data, weight, **attrs)
#     return None
