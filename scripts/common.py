"""Common utility for scripts"""
import argparse
import math
import os
import re
import time
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import topi
import tvm
from tvm import te
from tvm.ansor import (LogReader, make_workload_key_func,
                       register_auto_scheduler_workload_func,
                       write_measure_records_to_file)
from tvm.contrib import ndk, util

############################################################
######################  Test Workloads  ####################
############################################################

@register_auto_scheduler_workload_func
def min_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = topi.min(A, axis=1)

    return [A, B]

@register_auto_scheduler_workload_func
def argmin_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = topi.argmin(A, axis=1)

    return [A, B]

@register_auto_scheduler_workload_func
def softmax_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = topi.nn.softmax(A, axis=1)

    return [A, B]

@register_auto_scheduler_workload_func
def norm_bmn(B, M, N):
    A = te.placeholder((B, M, N), name='A')
    i = te.reduce_axis((0, M))
    j = te.reduce_axis((0, N))
    C = te.compute((B,), lambda b: te.sum(A[b][i][j] * A[b][i][j], axis=[i, j]), name='C')
    D = te.compute((B,), lambda b: te.sqrt(C[b]), name='D')

    return [A, D]

@register_auto_scheduler_workload_func
def add_mn(M, N):
    A = te.placeholder((M, N), name='A')
    B = te.placeholder((M, N), name='B')
    C = te.compute((M, N), lambda i, j: A[i][j] + B[i][j], name='C')

    return [A, B, C]

@register_auto_scheduler_workload_func
def matmul_nkkm(N, M, K, in_type='float32', out_type='float32',
                tensor_core_support=False):
    A = te.placeholder((N, K), name='A', dtype=in_type)
    B = te.placeholder((K, M), name='B', dtype=in_type)
    k = te.reduce_axis((0, K), name='k')
    if in_type == out_type:
        if not (in_type == 'float16' and out_type == 'float16'):
            tensor_core_support = False
        C = te.compute((N, M),
                        lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]),
                        name='C',
                        attrs={"auto_scheduler_tensor_core_support": "True" if tensor_core_support else "False"})
    else:
        if not ((in_type == 'float16' and out_type == 'float32') or \
                (in_type == 'int8' and out_type == 'int32')):
            tensor_core_support = False
        C = te.compute((N, M),
                        lambda i, j: te.sum(A[i][k].astype(out_type) * B[k][j].astype(out_type),
                                             axis=[k]),
                        name='C',
                        attrs={"auto_scheduler_tensor_core_support": "True" if tensor_core_support else "False"})

    return [A, B, C]

@register_auto_scheduler_workload_func
def dense_layer(batch, in_dim, out_dim):
    A = te.placeholder((batch, in_dim), name='A')
    B = te.placeholder((out_dim, in_dim), name='B')
    k = te.reduce_axis((0, in_dim), name='k')
    C = te.compute((batch, out_dim), lambda i, j: te.sum(A[i][k] * B[j][k], axis=[k]), name='C')

    return [A, B, C]

@register_auto_scheduler_workload_func
def max_pool_2d_nchw(N, C, H, W):
    data = te.placeholder((N, C, H, W), name='data')
    out = topi.nn.pool(data, (2, 2), (1, 1), (0, 0, 0, 0), pool_type='max', ceil_mode=True,
                       layout="NCHW", count_include_pad=True)

    return [data, out]

@register_auto_scheduler_workload_func
def add_min_relu(M, N):
    A = te.placeholder((M, N), name='A')
    B = te.placeholder((M, N), name='B')
    C = topi.add(A, B)
    D = topi.min(C, axis=1)
    out = topi.nn.relu(D)
    return [A, B, out]

@register_auto_scheduler_workload_func
def conv2d_relu_softmax_min(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    relu = topi.nn.relu(conv)
    softmax = topi.nn.softmax(relu, axis=1)
    out = topi.min(softmax, axis=1)

    return [data, kernel, out]

@register_auto_scheduler_workload_func
def conv2d_nchw_bias(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
    bias = te.placeholder((CO, 1, 1), name='bias')
    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    #out = topi.nn.relu(conv)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

def conv2d_nhwc_without_layout_rewrite(Input, Filter, stride, padding, dilation, out_dtype='float32'):
    """A copy of `topi.nn.conv2d_nhwc` but without the 'layout_free` attribute.
    We use this in single op and subgraph evaluation because we don't want to introduce graph level optimization.
    """
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape
    if len(Filter.shape) == 10:
        kernel_h = Filter.shape[2] * Filter.shape[6]
        kernel_w = Filter.shape[3] * Filter.shape[7]
        channel = Filter.shape[4] * Filter.shape[8]
        num_filter = Filter.shape[0] * Filter.shape[1] * Filter.shape[5] * Filter.shape[9]
        #Filter = te.placeholder([kernel_h, kernel_w, channel, num_filter], Filter.dtype, Filter.name)
    elif len(Filter.shape) == 11:
        kernel_h = Filter.shape[3] * Filter.shape[7]
        kernel_w = Filter.shape[4] * Filter.shape[8]
        channel = Filter.shape[5] * Filter.shape[9]
        num_filter = Filter.shape[0] * Filter.shape[1] * Filter.shape[2] * Filter.shape[6] * Filter.shape[10]
    else:
        kernel_h, kernel_w, channel, num_filter = Filter.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = topi.nn.util.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = topi.util.simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = topi.util.simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = topi.nn.pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            PaddedInput[nn, yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype)
            , axis=[ry, rx, rc]),
        name="Conv2dOutput", tag="conv2d_nhwc")
    return Output


@register_auto_scheduler_workload_func
def conv2d_nhwc_bias_with_rewrite(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, CO), name='kernel')
    bias = te.placeholder((CO, ), name='bias')
    conv = topi.nn.conv2d_nhwc(data, kernel, strides, padding, dilation)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

@register_auto_scheduler_workload_func
def depthwise_conv2d_nhwc_bias_with_rewrite(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, 1), name='kernel')
    bias = te.placeholder((CO, ), name='bias')
    conv = topi.nn.depthwise_conv2d_nhwc(data, kernel, strides, padding, dilation)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]

@register_auto_scheduler_workload_func
def conv2d_nhwc_bias(N, H, W, CI, CO, KH, KW, strides, padding, dilation):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((KH, KW, CI, CO), name='kernel')
    bias = te.placeholder((CO, ), name='bias')
    conv = conv2d_nhwc_without_layout_rewrite(data, kernel, strides, padding, dilation)
    out = topi.add(conv, bias)
    return [data, kernel, bias, out]


@register_auto_scheduler_workload_func
def conv2d_nchw_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation=1):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, kernel_size, kernel_size), name='kernel')
    bias = te.placeholder((CO, 1, 1), name='bias')
    bn_scale = te.placeholder((CO, 1, 1), name='bn_scale')
    bn_offset = te.placeholder((CO, 1, 1), name='bn_offset')

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1

    conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation)
    conv = te.compute((N, CO, OH, OW),
                       lambda i, j, k, l: conv[i, j, k, l] + bias[j, 0, 0],
                       name='bias_add')
    conv = te.compute((N, CO, OH, OW),
                       lambda i, j, k, l: conv[i, j, k, l] * bn_scale[j, 0, 0],
                       name='bn_mul')
    conv = te.compute((N, CO, OH, OW),
                       lambda i, j, k, l: conv[i, j, k, l] + bn_offset[j, 0, 0],
                       name='bn_add')
    out = topi.nn.relu(conv)

    return [data, kernel, bias, bn_offset, bn_scale, out]

@register_auto_scheduler_workload_func
def conv2d_nhwc_bn_relu(N, H, W, CI, CO, kernel_size, strides, padding, dilation=1):
    data = te.placeholder((N, H, W, CI), name='data')
    kernel = te.placeholder((kernel_size, kernel_size, CI, CO), name='kernel')
    bias = te.placeholder((CO,), name='bias')
    bn_scale = te.placeholder((CO,), name='bn_scale')
    bn_offset = te.placeholder((CO,), name='bn_offset')

    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1

    conv = conv2d_nhwc_without_layout_rewrite(data, kernel, strides, padding, dilation)
    conv = te.compute((N, OH, OW, CO),
                       lambda i, j, k, l: conv[i, j, k, l] + bias[l],
                       name='bias_add')
    conv = te.compute((N, OH, OW, CO),
                       lambda i, j, k, l: conv[i, j, k, l] * bn_scale[l],
                       name='bn_mul')
    conv = te.compute((N, OH, OW, CO),
                       lambda i, j, k, l: conv[i, j, k, l] + bn_offset[l],
                       name='bn_add')
    out = topi.nn.relu(conv)

    return [data, kernel, bias, bn_offset, bn_scale, out]

resnet_conv2d_configs = {
    # format : N, H, W, CI, CO, KH, KW, strides, padding, dilation
    '18': [
        (1, 224, 224, 3, 64, 7, 7, (2, 2), (3, 3), (1, 1)),
        (1, 56, 56, 64, 128, 3, 3, (2, 2), (1, 1), (1, 1)),
        (1, 56, 56, 64, 128, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 56, 56, 64, 64, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 128, 256, 3, 3, (2, 2), (1, 1), (1, 1)),
        (1, 28, 28, 128, 256, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 14, 14, 256, 512, 3, 3, (2, 2), (1, 1), (1, 1)),
        (1, 14, 14, 256, 512, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)),
    ],
    '50': [
        (1, 224, 224, 3, 64, 7, 7, (2, 2), (3, 3), (1, 1)),
        (1, 56, 56, 256, 512, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 56, 56, 256, 128, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 56, 56, 256, 64, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 56, 56, 64, 256, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 56, 56, 64, 64, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 512, 1024, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 28, 28, 512, 256, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 28, 28, 512, 128, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 128, 512, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 14, 14, 1024, 2048, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 14, 14, 1024, 512, 1, 1, (2, 2), (0, 0), (1, 1)),
        (1, 14, 14, 1024, 256, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 14, 14, 256, 1024, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1), (1, 1)),
        (1, 7, 7, 2048, 512, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 7, 7, 512, 2048, 1, 1, (1, 1), (0, 0), (1, 1)),
        (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)),
    ],
}

# number of appearance for all conv2ds in resnet
resnet_conv2d_weights = {
    '18': [1, 1, 1, 4, 1, 1, 1, 3, 1, 1, 3, 3],
    '50': [1, 1, 1, 2, 4, 3, 1, 1, 1, 3, 4, 4, 1, 1, 5, 6, 6, 2, 3, 3],
}


def parse_workload_name(name: str) -> List[str]:
    """Parse workload name with wildcard character and abbreviation to standard names"""
    if name.startswith('matmul-'):  # e.g. matmul-512, matmul-1024, matmul-+
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [256, 512, 1024]
        else:
            cfg_list = [N]
        return ["matmul-%s" % x for x in cfg_list]
    elif name.startswith('dense-'):  # e.g. dense-1-512-1024, dense-16-512-512
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = ["1-512-512", "16-512-512"]
        else:
            cfg_list = [N]
        return ["dense-%s" % x for x in cfg_list]
    elif name.startswith('min-'):  # e.g. min-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["min-%s" % x for x in cfg_list]
    elif name.startswith('argmin-'):  # e.g. argmin-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["argmin-%s" % x for x in cfg_list]
    elif name.startswith('softmax-'):  # e.g. softmax-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["softmax-%s" % x for x in cfg_list]
    elif name.startswith('add-'):  # e.g. add-4096
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["add-%s" % x for x in cfg_list]
    elif name.startswith('norm-'):  # e.g. norm-1024
        N = name.split('-', maxsplit=1)[1]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["norm-%s" % x for x in cfg_list]
    elif name.startswith('add-min-relu'):  # e.g. add-min-relu-4096
        N = name.split('-', maxsplit=3)[3]
        if N == '+':
            cfg_list = [4096, 8192, 16384]
        else:
            cfg_list = [N]
        return ["add-min-relu-%s" % x for x in cfg_list]
    elif name.startswith('nhwc-resnet-'):  # e.g.  nhwc-resnet-50.C1
        res = re.match(r'nhwc-resnet-(\d+).C([\d\+]+)(.B(\d+))?', name)
        n_layers = res.group(1)
        if res.group(2) == '+':
            idx_list = range(len(resnet_conv2d_configs[n_layers]))
        else:
            idx_list = [int(res.group(2))]

        batch_size = 1 if res.group(4) is None else int(res.group(4))
        return ['nhwc-resnet-%s.C%d.B%d' % (n_layers, i, batch_size) for i in idx_list]
    elif name.startswith('resnet-'):  # e.g.  resnet-50.C1, resnet-50.C1.B2, resnet-50.C+.B2
        res = re.match(r'resnet-(\d+).C([\d\+]+)(.B(\d+))?', name)
        n_layers = res.group(1)
        if res.group(2) == '+':
            idx_list = range(len(resnet_conv2d_configs[n_layers]))
        else:
            idx_list = [int(res.group(2))]

        batch_size = 1 if res.group(4) is None else int(res.group(4))
        return ['resnet-%s.C%d.B%d' % (n_layers, i, batch_size) for i in idx_list]
    elif name in ['conv2d-bn-relu', 'conv2d-relu-softmax-min', 'max-pool-2d', 'conv2d-rewrite', 'depthwise-conv2d-rewrite']:
        return [name]
    else:
        raise ValueError("Invalid workload " + name)


def get_workload_keys(name: str) -> List[str]:
    """Parse workload name and return the workload keys"""
    normalized_names = parse_workload_name(name)

    ret = []
    for name in normalized_names:
        if name.startswith('matmul-'):
            name_split = name.split('-')
            in_type = out_type = 'float32'
            tensor_core_support = False
            if len(name_split) == 2:    # e.g. matmul-512
                N = K = M = int(name_split[1])
            elif len(name_split) == 4:  # e.g. matmul-32-256-512
                N = int(name_split[1])
                K = int(name_split[2])
                M = int(name_split[3])
            elif len(name_split) == 6:  # e.g. matmul-32-512-512-float16-float32
                N = int(name_split[1])
                K = int(name_split[2])
                M = int(name_split[3])
                in_type = name_split[4]
                out_type = name_split[5]
            elif len(name_split) == 7:  # e.g. matmul-32-512-512-float16-float32-tc
                N = int(name_split[1])
                K = int(name_split[2])
                M = int(name_split[3])
                in_type = name_split[4]
                out_type = name_split[5]
                tensor_core_support = name_split[6] == "tc"
            else:
                raise ValueError("Invalid matmul workload")
            ret.append(make_workload_key_func(matmul_nkkm,
                                              (N, M, K, in_type, out_type, tensor_core_support)))
        elif name.startswith('dense-'):  # e.g. dense-1-512-1024, dense-16-512-512
            name_split = name.split('-')
            assert len(name_split) == 4
            batch = int(name_split[1])
            in_dim = int(name_split[2])
            out_dim = int(name_split[3])
            ret.append(make_workload_key_func(dense_layer, (batch, in_dim, out_dim)))
        elif name.startswith('min-'):  # e.g. min-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                M = 64
                N = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid min workload")
            ret.append(make_workload_key_func(min_mn, (M, N)))
        elif name.startswith('argmin-'):  # e.g. argmin-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                M = 64
                N = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid argmin workload")
            ret.append(make_workload_key_func(argmin_mn, (M, N)))
        elif name.startswith('softmax-'):  # e.g. softmax-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                M = 64
                N = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid softmax workload")
            ret.append(make_workload_key_func(softmax_mn, (M, N)))
        elif name.startswith('add-min-relu'):  # e.g. add-min-relu-4096
            name_split = name.split('-')
            if len(name_split) == 4:
                M = 64
                N = int(name_split[3])
            elif len(name_split) == 5:
                M = int(name_split[3])
                N = int(name_split[4])
            else:
                raise ValueError("Invalid workload")
            ret.append(make_workload_key_func(add_min_relu, (M, N)))
        elif name.startswith('add-'):  # e.g. add-4096
            name_split = name.split('-')
            if len(name_split) == 2:
                N = M = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid add workload")
            ret.append(make_workload_key_func(add_mn, (M, N)))
        elif name.startswith('norm-'):  # e.g. norm-4096
            name_split = name.split('-')
            B = 2
            if len(name_split) == 2:
                N = M = int(name_split[1])
            elif len(name_split) == 3:
                M = int(name_split[1])
                N = int(name_split[2])
            else:
                raise ValueError("Invalid norm workload")
            ret.append(make_workload_key_func(norm_bmn, (B, M, N)))
        elif name.startswith('nhwc-resnet-'):  # e.g.  nhwc-resnet-50.C1.B2
            res = re.match(r'nhwc-resnet-(\d+).C(\d+).B(\d+)', name)
            n_layers = res.group(1)
            idx = int(res.group(2))
            batch_size = 1 if res.group(3) is None else int(res.group(3))
            args = list(resnet_conv2d_configs[n_layers][idx])
            args[0] = batch_size
            ret.append(make_workload_key_func(conv2d_nhwc_bias, args))
        elif name.startswith('resnet-'):  # e.g.  resnet-50.C1.B2
            res = re.match(r'resnet-(\d+).C(\d+).B(\d+)', name)
            n_layers = res.group(1)
            idx = int(res.group(2))
            batch_size = 1 if res.group(3) is None else int(res.group(3))
            args = list(resnet_conv2d_configs[n_layers][idx])
            args[0] = batch_size
            ret.append(make_workload_key_func(conv2d_nchw_bias, args))
        elif name == 'max-pool-2d':
            return [make_workload_key_func(max_pool_2d_nchw, (2, 512, 7, 7))]
        elif name == 'conv2d-bn-relu':
            return [make_workload_key_func(conv2d_nhwc_bn_relu,
                                           (1, 7, 7, 512, 512, 3, 1, 1, 1)) ]
        elif name == 'conv2d-rewrite':
            return [ make_workload_key_func(conv2d_nhwc_bias_with_rewrite,
                                            (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)))]
        elif name == 'depthwise-conv2d-rewrite':
            return [ make_workload_key_func(depthwise_conv2d_nhwc_bias_with_rewrite,
                                            (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)))]
        elif name == 'conv2d-relu-softmax-min':
            return [make_workload_key_func(conv2d_relu_softmax_min,
                                           (1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1), (1, 1)))]
        else:
            raise ValueError("Invalid workload " + name)

    return ret


def get_workload_weights(name: str) -> List[float]:
    """Return weights for workload name"""
    if name.startswith('resnet-'):
        res = re.match(r'resnet-(\d+).C+', name)
        n_layers = res.group(1)
        return np.array(resnet_conv2d_weights[n_layers])
    else:
        return np.ones(len(get_workload_keys(name)))


############################################################
######################  Measure Tools   ####################
############################################################


def measure_schedule(s,
                     bufs,
                     target,
                     target_host=None,
                     remote=None,
                     ndk_cc=None,
                     number=10,
                     repeat=3,
                     min_repeat_ms=500):
    """Measure the time cost of a schedule"""
    func = tvm.build(s, bufs, target=target, target_host=target_host)
    if remote:
        ctx = remote.context(str(target), 0)
        temp = util.tempdir()
        remote_path = temp.relpath("tmp_deploy_lib.so")
        os.environ['TVM_NDK_CC'] = ndk_cc
        func.export_library(remote_path, ndk.create_shared)
        remote.upload(remote_path)
        func = remote.load_module("tmp_deploy_lib.so")
    else:
        ctx = tvm.context(str(target), 0)

    if os.environ.get('TVM_AUTO_CACHE_FLUSH', '0') == '1':
        min_repeat_ms = 0
        number = 1

    time_f = func.time_evaluator(func.entry_name,
                                 ctx,
                                 number=number,
                                 repeat=repeat,
                                 min_repeat_ms=min_repeat_ms)

    np_args = [np.ones(topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs]
    args = [tvm.nd.array(x, ctx=ctx) for x in np_args]
    ctx.sync()

    costs = time_f(*args).results

    return costs

def check_correctness(s, bufs, s_ref, buf_ref, target, target_host=None, remote=None, ndk_cc=None):
    """Check the correctness of a schedule against a reference schedule"""
    func = tvm.build(s, bufs, target=target, target_host=target_host)
    func_ref = tvm.build(s_ref, buf_ref, target='llvm')

    if remote:
        raise NotImplemented
    else:
        ctx = tvm.context(str(target), 0)
        ctx_ref = tvm.cpu()

    np_args = [np.ones(topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs]
    args = [tvm.nd.array(x, ctx=ctx) for x in np_args]
    args_ref = [tvm.nd.array(x, ctx=ctx_ref) for x in np_args]
    ctx.sync()

    func(*args)
    func_ref(*args_ref)

    for arr, arr_ref in zip(args, args_ref):
        np.testing.assert_allclose(arr.asnumpy(), arr_ref.asnumpy())


############################################################
#####################  Other Utilities  ####################
############################################################


def geomean(xs):
    """Compute geometric mean"""
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


global last_tic
last_tic = None


def PRINT_TIME(msg):
    """Print time interval between differnt calls. This is for debug so we make the name letters capital"""
    global last_tic
    now = time.time()

    if last_tic is None:
        last_tic = now

    print(msg, now - last_tic)
    last_tic = now


############################################################
######################  I/O Utilities  #####################
############################################################

# The format for a line in resulst file
BenchmarkRecord = namedtuple("BenchmarkRecord", [
    'device', 'backend', 'workload_type', 'workload_name', 'library', 'algorithm', 'value',
    'time_stamp'
])


class BaselineDatabase:
    """A class for query records in baseline database"""
    def __init__(self, filename):
        self.filename = filename

        self.lines = []
        for line in open(filename):
            if line.startswith('#') or line.isspace():
                continue
            self.lines.append(line.split('\t'))

    def filter_records(self, devices=None, backends=None, wkl_names=None, libraries=None):
        ret = []
        for line in self.lines:
            line = BenchmarkRecord(*line)

            if devices is not None and line.device not in devices:
                continue
            if backends is not None and line.backend not in backends:
                continue
            if wkl_names is not None and line.workload_name not in wkl_names:
                continue
            if libraries is not None and line.library not in libraries:
                continue

            ret.append(line)
        return ret

    def get_data_dict(self, device, target, wkl_names) -> Tuple[Dict, List]:
        """Return a data dict s.t.  data[wkl][library] = cost"""
        data = defaultdict(lambda: defaultdict(lambda: 1e10))

        all_libraries = set()

        if "cpu" in target.keys:
            backends = ['cpu']
        elif "gpu" in target.keys:
            backends = ['gpu']
        else:
            raise ValueError("Invalid target: " + target)

        # Read costs for baselines
        records = self.filter_records(devices=[device], backends=backends, wkl_names=wkl_names)
        for record in records:
            # use min over (possible) multiple algorithms
            all_libraries.add(record.library)
            data[record.workload_name][record.library] = \
                min(data[record.workload_name][record.library],
                    np.mean(eval(record.value)['costs']))

        return data, list(all_libraries)


class LogFileDatabase:
    """A class for indexing best records in a log file"""
    def __init__(self, filename: str, n_lines: int = -1):
        inputs, results = LogReader(filename).read_lines(n_lines)

        # best records, search by (target_key, workload_key).  e.g. ('gpu', 'conv2d...')
        self.best_by_targetkey = {}

        # best according to (model, workload_key).  e.g. ('1080ti', 'conv2d...'))
        self.best_by_model = {}

        # find best records and build the index
        for inp, res in zip(inputs, results):
            if res.error_no != 0:
                continue

            # use target keys in tvm target system as key to build best map
            for target_key in inp.task.target.keys:
                key = (target_key, inp.task.workload_key)
                if key not in self.best_by_targetkey:
                    self.best_by_targetkey[key] = (inp, res)
                else:
                    _, other_res = self.best_by_targetkey[key]
                    if np.mean([x.value for x in other_res.costs]) > \
                            np.mean([x.value for x in res.costs]):
                        self.best_by_targetkey[key] = (inp, res)

            # use model as key to build best map
            key = (inp.task.target.model, inp.task.workload_key)
            if key not in self.best_by_model:
                if inp.task.target.model != 'unknown':
                    self.best_by_model[key] = (inp, res)
            else:
                _, other_res = self.best_by_model[key]
                if np.mean([x.value for x in other_res.costs]) > \
                        np.mean([x.value for x in res.costs]):
                    self.best_by_model[key] = (inp, res)

    def write_best(self, filename: str):
        best_records = list(self.best_by_targetkey.values())
        inputs = [x[0] for x in best_records]
        results = [x[1] for x in best_records]
        write_measure_records_to_file(filename, inputs, results)


############################################################
######################  Plot Utilities  ####################
############################################################

def max_curve(raw_curve):
    """Return b[i] = max(a[:i]) """
    ret = []
    cur_max = -np.inf
    for x in raw_curve:
        cur_max = max(cur_max, x)
        ret.append(cur_max)
    return ret

def min_curve(raw_curve):
    """Return b[i] = min(a[:i]) """
    ret = []
    cur_min = np.inf
    for x in raw_curve:
        cur_min = min(cur_min, x)
        ret.append(cur_min)
    return ret

def mean_curve(raw_curve, window_size=None):
    """Return b[i] = mean(a[:i]) """
    ret = []
    mean = 0
    if window_size is None:
        for i, x in enumerate(raw_curve):
            mean = (mean * i + x) / (i + 1)
            ret.append(mean)
    else:
        for i, x in enumerate(raw_curve):
            if i >= window_size:
                mean = (mean * window_size + x - raw_curve[i - window_size]) / window_size
            else:
                mean = (mean * i + x) / (i + 1)
            ret.append(mean)
    return ret


def enhance_color(color, h=1, l=1, s=1):
    """Make color looks better for pyplot"""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    h, l, s = h * c[0], l * c[1], s * c[2]
    h, l, s = [max(min(x, 1), 0) for x in [h, l, s]]

    return colorsys.hls_to_rgb(h, l, s)


method_color_dict = {
    'ours': 'C0',
    'AutoTVM': 'C1',

    'tensorflow': 'C2',
    'tensorflow-tensorrt': 'C9',
    'tflite': 'C2',

    'pytorch': enhance_color('C3', l=1.1, s=0.9),

    'FlexTensor': enhance_color('C5'),
    'halide': enhance_color('teal', l=1.25),

    'Limit space': 'C7',
    'No fine-tuning': 'C8',
    'No task scheduler': 'C1',
}

def method2color(method):
    if '-batch-' in method:
        method, batch_size = method.split('-batch-')
        #return enhance_color(method_color_dict[method], s=1.1, l=1.5)
        return method_color_dict[method]
    else:
        return method_color_dict[method]

method_order_list = [
    'pytorch', 'tensorflow', 'tensorflow-xla', 'tensorflow-tensorrt',
    'tflite', 'halide', 'FlexTensor',  'AutoTVM',

    'Limit space', 'No fine-tuning',
    'ours',
]

def method2order(method):
    if '-batch-' in method:
        method, batch_size = method.split('-batch-')
        batch_size = int(batch_size)
        return method_order_list.index(method) + batch_size / 100
    else:
        return method_order_list.index(method)

show_name_replace_dict = {
    'pytorch': "PyTorch",
    'tensorflow-tensorrt': 'TensorRT-TF',
    'tensorflow': 'TensorFlow',
    'tflite': 'TensorFlow Lite',
    'halide': 'Halide',

    'ours': 'Ansor (ours)',
    'batch-16': 'batch',

    'resnet_50': 'ResNet-50',
    'mobilenet_v2': 'Mobilenet V2',
    'resnet_18_3d': '3D-ResNet',
    'dcgan': 'DCGAN',
    'dqn': 'DQN',
    'bert': 'BERT',
}

def show_name(name):
    #    if name.startswith('resnet-'):
    #        return name.split('.')[1]
    for key, value in show_name_replace_dict.items():
        name = name.replace(key, value)

    return name

def draw_grouped_bar_chart(data, baseline='pytorch', output='out.png',
                           yscale_log=False, yticks=None, y_max=None,
                           legend_bbox_to_anchor=None, legend_nrow=None,
                           figure_size=None, figax=None, draw_ylabel=True, draw_legend=True):
    width = 1
    gap = 1.5
    fontsize = 19
    xticks_font_size = fontsize - 2

    figure_size = figure_size or (11, 4)
    legend_bbox_to_anchor = legend_bbox_to_anchor or (0.45, 1.35)

    all_methods = set()
    legend_set = {}

    if figax is None:
        fig, ax = plt.subplots()
        axes = []
        axes.append(ax)
    else:
        ax = figax

    x0 = 0
    xticks = []
    xlabels = []

    workloads = list(data.keys())
    for wkl in workloads:
        ys = []
        colors = []

        methods = list(data[wkl].keys())

        if baseline in data[wkl]:
            baseline_cost = data[wkl][baseline]
        else:
            # normalize to best library
            baseline_cost = 1e10
            for method in methods:
                if data[wkl][method] < baseline_cost:
                    baseline_cost = data[wkl][method]

        methods.sort(key=lambda x: method2order(x))
        for method in methods:
            relative_speedup = baseline_cost / data[wkl][method]
            if yticks is None:
                ys.append(relative_speedup)
            else:
                ys.append(max(relative_speedup, yticks[0] * 1.1))
            colors.append(method2color(method))

        # draw the bars
        xs = np.arange(x0, x0 + len(ys))
        bars = ax.bar(xs, ys, width=width, color=colors)

        for method, bar_obj in zip(methods, bars):
            all_methods.add(method)
            if method not in legend_set:
                legend_set[method] = bar_obj

        # tick and label
        x0 += len(ys) + gap

        xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
        xlabels.append(show_name(wkl))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=xticks_font_size)
        plt.tick_params(axis='x', which='both', bottom='off', top='off')

        if draw_ylabel is True:
            ax.set_ylabel('Relative Speedup', fontsize=fontsize)
        elif isinstance(draw_ylabel, str):
            ax.set_ylabel(draw_ylabel, fontsize=fontsize)

        if yscale_log:
            ax.set_yscale('log', basey=2)
        if yticks is not None:
            ax.set_yticks(yticks)
        if y_max:
            ax.set_ylim(top=y_max)

        from matplotlib.ticker import FormatStrFormatter
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.grid(linewidth=0.4, linestyle='dotted') # draw grid line
        ax.set_axisbelow(True)  # grid lines are behind the rest
        ax.tick_params(bottom=False, top=False, right=False)

    # put legend outside the plot
    all_methods = list(all_methods)
    all_methods.sort(key=lambda x : method2order(x))

    if draw_legend:
        legend_nrow = legend_nrow or 2
        ncol = (len(all_methods) + legend_nrow - 1)// legend_nrow
        ax.legend([legend_set[x] for x in all_methods],
                  [show_name(x) for x in all_methods],
                  fontsize=fontsize-1,
                  loc='upper center',
                  bbox_to_anchor=legend_bbox_to_anchor,
                  ncol=ncol,
                  handlelength=1.0,
                  handletextpad=0.5,
                  columnspacing=1.1)

    if figax is None:
        fig.set_size_inches(figure_size)
        fig.savefig(output, bbox_inches='tight')
        print("Output the plot to %s" % output)


def to_str_round(x, decimal=6):
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) or isinstance(x, np.ndarray):
        return "[" + ", ".join([to_str_round(y, decimal=decimal)
                                for y in x]) + "]"
    if isinstance(x, dict):
        return str({k: eval(to_str_round(v)) for k, v in x.items()})
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        format_str = "%%.%df" % decimal
        return format_str % x
    raise ValueError("Invalid value: " + str(x))

