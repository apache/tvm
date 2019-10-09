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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D alter op and legalize functions for x86"""

import logging

import tvm
from tvm import relay
from tvm import autotvm
from .conv2d import _get_default_config
from .conv2d_int8 import _is_int8_hw_support, _get_default_config_int8
from ..util import get_const_tuple, get_shape
from ..nn import conv2d_legalize
from ..nn.conv2d import conv2d, conv2d_NCHWc, conv2d_NCHWc_int8, conv2d_alter_layout
from ..nn.depthwise_conv2d import depthwise_conv2d_NCHWc, depthwise_conv2d_nchw

logger = logging.getLogger('topi')

@conv2d_alter_layout.register("cpu")
def _alter_conv2d_layout(attrs, inputs, tinfo, F):
    # Parse the attributes.
    groups = attrs.get_int("groups")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    out_dtype = attrs["out_dtype"]
    layout_name = 'layout' if F.__name__ == 'nnvm.symbol' else 'data_layout'
    data_layout = attrs[layout_name]
    kh, kw = attrs.get_int_tuple("kernel_size")

    data_tensor, kernel_tensor = tinfo[0], tinfo[1]
    if attrs[layout_name] == 'NHWC' and attrs['kernel_layout'] == 'HWIO':
        batch_size, height, width, in_channel = get_const_tuple(data_tensor.shape)
        kh, kw, _, out_channel = get_const_tuple(kernel_tensor.shape)
    elif attrs[layout_name] == 'NCHW' and attrs['kernel_layout'] == 'OIHW':
        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        out_channel, _, kh, kw = get_const_tuple(kernel_tensor.shape)
    else:
        return None

    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype
    out_dtype = data_dtype if out_dtype in ("same", "") else out_dtype

    # Check if depthwise.
    kshape = get_shape(kernel_tensor.shape, attrs["kernel_layout"], "OIHW")
    is_depthwise = groups == kshape[0] and kshape[1] == 1

    # Save the input exprs.
    copy_inputs = [s for s in inputs]

    # Set the new attrs
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['channels'] = out_channel

    # Return if the groups is not 1 and depthwise.
    if groups != 1 and not is_depthwise:
        return None

    # Set workload. Config update.
    dispatch_ctx = autotvm.task.DispatchContext.current
    target = tvm.target.current_target()

    if is_depthwise:
        workload = autotvm.task.args_to_workload(
            [data_tensor, kernel_tensor, strides, padding, dilation, out_dtype],
            depthwise_conv2d_nchw)
    else:
        workload = autotvm.task.args_to_workload(
            [data_tensor, kernel_tensor, strides, padding, dilation, data_layout, out_dtype],
            conv2d)

    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:
        if _is_int8_hw_support(data_dtype, kernel_dtype):
            _get_default_config_int8(cfg, data_tensor, kernel_tensor, strides, padding, out_dtype,
                                     is_depthwise, data_layout)
        else:
            _get_default_config(cfg, data_tensor, kernel_tensor, strides, padding, out_dtype,
                                is_depthwise, data_layout)

    # Get the tiling parameters to set the layout names.
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    new_attrs[layout_name] = 'NCHW%dc' % ic_bn
    new_attrs['out_layout'] = 'NCHW%dc' % oc_bn
    new_data = tvm.placeholder((batch_size, in_channel//ic_bn, height, width, ic_bn),
                               dtype=data_dtype)

    if is_depthwise and data_layout == 'NCHW' and attrs['kernel_layout'] == 'OIHW':
        new_attrs['kernel_layout'] = 'OIHW1i%do' % oc_bn
        # Store altered operator's config
        new_kernel = tvm.placeholder((out_channel//oc_bn, 1, kh, kw, 1, oc_bn), dtype=kernel_dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_attrs[layout_name],
             new_attrs['out_layout'], out_dtype], depthwise_conv2d_NCHWc)
        dispatch_ctx.update(target, new_workload, cfg)
        if F.__name__ == 'nnvm.symbol':
            logging.warning("Use native layout for depthwise convolution on NNVM.")
            return None
        return F.nn.contrib_depthwise_conv2d_nchwc(*copy_inputs, **new_attrs)

    if _is_int8_hw_support(data_dtype, kernel_dtype):
        # Convert kernel data layout from 4D to 7D
        n_elems = 4
        data_expr, kernel_expr = inputs
        if attrs['kernel_layout'] == 'HWIO':
            kernel_IHWO = F.transpose(kernel_expr, axes=(2, 0, 1, 3))
        elif attrs['kernel_layout'] == 'OIHW':
            kernel_IHWO = F.transpose(kernel_expr, axes=(1, 2, 3, 0))
        else:
            return None

        kernel_IHWOo = F.reshape(kernel_IHWO, (in_channel, kh, kw, out_channel//oc_bn, oc_bn))
        kernel_OHWoI = F.transpose(kernel_IHWOo, axes=(3, 1, 2, 4, 0))
        kernel_OHWoIi = F.reshape(kernel_OHWoI, (out_channel//oc_bn, kh, kw, oc_bn,
                                                 in_channel//ic_bn, ic_bn))
        kernel_OHWoIie = F.reshape(kernel_OHWoIi, (out_channel//oc_bn, kh, kw, oc_bn,
                                                   in_channel//ic_bn, ic_bn//n_elems, n_elems))
        kernel_OIHWioe = F.transpose(kernel_OHWoIie, axes=(0, 4, 1, 2, 5, 3, 6))
        copy_inputs = [data_expr, kernel_OIHWioe]

        # Store altered operator's config. New kernel layout OIHWio4
        new_kernel = tvm.placeholder((out_channel // oc_bn,
                                      in_channel // ic_bn,
                                      kh,
                                      kw,
                                      ic_bn // n_elems,
                                      oc_bn,
                                      n_elems), dtype=kernel_dtype)

        new_workload = autotvm.task.args_to_workload([new_data,
                                                      new_kernel,
                                                      strides,
                                                      padding,
                                                      dilation,
                                                      new_attrs[layout_name],
                                                      new_attrs['out_layout'],
                                                      out_dtype],
                                                     conv2d_NCHWc_int8)
        dispatch_ctx.update(target, new_workload, cfg)
        if F.__name__ == 'nnvm.symbol':
            logging.warning("Use native layout for int8 convolution on NNVM.")
            return None
        return F.nn.contrib_conv2d_nchwc_int8(*copy_inputs, **new_attrs)

    # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
    new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)
    # Store altered operator's config
    new_kernel = tvm.placeholder((out_channel//oc_bn, in_channel//ic_bn,
                                  kh, kw, ic_bn, oc_bn), dtype=kernel_tensor.dtype)
    new_workload = autotvm.task.args_to_workload(
        [new_data, new_kernel, strides, padding, dilation, new_attrs[layout_name],
         new_attrs['out_layout'], out_dtype], conv2d_NCHWc)
    dispatch_ctx.update(target, new_workload, cfg)

    if F.__name__ == 'nnvm.symbol':
        return F.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
    return F.nn.contrib_conv2d_nchwc(*copy_inputs, **new_attrs)


@conv2d_legalize.register("cpu")
def _conv2d_legalize(attrs, inputs, arg_types):
    """Legalizes Conv2D op.

    Parameters
    ----------
    attrs : tvm.attrs.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """

    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]

    # Collect the output tensor.
    output_tensor = arg_types[2]

    # Legalize if the datatypes are suitable for fast Int8 instructions.  Int8 instructions require
    # input channel to be a multiple of 4 and output channels to be a multiple of 16. For input
    # channels, we pad both the inputs and weights input channels. For output channels, we pad the
    # weight and stride_slice the output.
    if _is_int8_hw_support(data_tensor.dtype, kernel_tensor.dtype):
        # Flags to remember if the expr is modified
        ic_modified = False
        oc_modified = False

        # Collect the input exprs.
        data, kernel = inputs

        # Find the value of input and output channel.
        in_channel = -1
        out_channel = -1
        if attrs['data_layout'] == 'NHWC' and attrs['kernel_layout'] == 'HWIO':
            in_channel = data_tensor.shape[3].value
            out_channel = kernel_tensor.shape[3].value
        elif attrs['data_layout'] == 'NCHW' and attrs['kernel_layout'] == 'OIHW':
            in_channel = data_tensor.shape[1].value
            out_channel = kernel_tensor.shape[0].value
        else:
            return None

        if in_channel % 4 != 0:
            new_in_channel = ((in_channel + 4) // 4) * 4
            diff = new_in_channel - in_channel
            if attrs['data_layout'] == 'NHWC' and attrs['kernel_layout'] == 'HWIO':
                data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
                kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, diff), (0, 0)))
                ic_modified = True
            elif attrs['data_layout'] == 'NCHW' and attrs['kernel_layout'] == 'OIHW':
                pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
                data = relay.nn.pad(data, pad_width=pad_width)
                kernel = relay.nn.pad(kernel, pad_width=pad_width)
                ic_modified = True
            else:
                return None

        new_out_channel = out_channel
        if out_channel % 16 != 0:
            new_out_channel = ((out_channel + 16) // 16) * 16
            diff = new_out_channel - out_channel
            if attrs['data_layout'] == 'NHWC' and attrs['kernel_layout'] == 'HWIO':
                kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
                oc_modified = True
            elif attrs['data_layout'] == 'NCHW' and attrs['kernel_layout'] == 'OIHW':
                kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
                oc_modified = True
            else:
                return None

        if not (ic_modified or oc_modified):
            return None

        if ic_modified and not oc_modified:
            return relay.nn.conv2d(data, kernel, **attrs)

        if oc_modified:
            new_attrs = {k: attrs[k] for k in attrs.keys()}
            new_attrs['channels'] = new_out_channel
            out = tvm.relay.nn.conv2d(data, kernel, **new_attrs)
            original_out_shape = [x.value for x in output_tensor.shape]
            return relay.strided_slice(out, begin=(0, 0, 0, 0), end=original_out_shape)
    return None
