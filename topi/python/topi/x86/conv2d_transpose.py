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
"""Conv2D Transpose schedule on x86"""
import tvm
from tvm import autotvm
from .. import generic
from ..util import get_const_tuple, traverse_inline
from ..nn import conv2d_transpose_nchw_preprocess, conv2d_transpose_nchw
from . import conv2d_avx_1x1, conv2d_avx_common
from .conv2d import _declaration_conv_impl, \
    _create_tuning_space as _create_tuning_space_conv2d, \
    _get_default_config as _get_default_config_conv2d


@autotvm.register_topi_compute(conv2d_transpose_nchw, 'cpu', ['direct'])
def _conv2d_transpose_nchw(cfg, data, kernel, strides, padding, out_dtype):
    data_pad, kernel_transform = \
        conv2d_transpose_nchw_preprocess(data, kernel, strides, padding, out_dtype)
    # reuse conv2d implementation
    _create_tuning_space_conv2d(cfg, data_pad, kernel_transform, strides=(1, 1), \
                                padding=(0, 0), dilation=(1, 1), layout="NCHW")
    if cfg.is_fallback:
        _get_default_config_conv2d(cfg, data_pad, kernel_transform, strides=(1, 1), \
                                   padding=(0, 0), out_dtype=out_dtype, layout='NCHW')
    return _declaration_conv_impl(cfg, data_pad, kernel_transform, strides=(1, 1), \
                                  padding=(0, 0), dilation=(1, 1), layout="NCHW", \
                                  out_dtype=out_dtype)


@autotvm.register_topi_schedule(generic.schedule_conv2d_transpose_nchw, 'cpu', ['direct'])
def _schedule_conv2d_transpose_nchw(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        # reuse conv2d schedule
        if 'conv2d_nchw' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            # retrieve data
            data_vec = conv_out.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            data_dilate = data_pad.op.input_tensors[0]
            s[data_dilate].compute_inline()
            # retrieve kernel
            kernel_vec = conv_out.op.input_tensors[1]
            kernel_transform = kernel_vec.op.input_tensors[0]
            s[kernel_transform].compute_inline()
            # call conv2d schedule
            _, _, kh, kw = get_const_tuple(kernel_transform.shape)
            is_kernel_1x1 = kh == 1 and kw == 1
            args = [s, cfg, data_dilate, data_pad, data_vec, kernel_vec, conv_out, output, outs[0]]
            if is_kernel_1x1:
                conv2d_avx_1x1._schedule_conv(*args)
            else:
                conv2d_avx_common._schedule_conv(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s
