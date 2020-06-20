"""Tune all workloads for single op & subgraph evaluation"""
import argparse
import logging
import random

import numpy as np

import tvm
from tvm import te, ansor
import topi
from topi.nn.winograd_util import winograd_transform_matrices
from topi.util import get_const_tuple

from common import measure_schedule, str2bool, norm_bmn, conv2d_nhwc_bn_relu, conv2d_nchw_bn_relu
from shape_configs import single_op_shape_dict, subgraph_shape_dict
from tune_test import tune_workloads_jointly, replay_workload, create_tune_option

# ========================== Single Ops ==========================

@ansor.register_workload_func
def batch_matmul_nkkm(B, N, M, K):
    X = te.placeholder((B, N, K), name='A')
    Y = te.placeholder((B, K, M), name='B')
    k = te.reduce_axis((0, K), name='k')
    Z = te.compute((B, N, M), lambda b, i, j: te.sum(X[b][i][k] * Y[b][k][j], axis=[k]), name='C')
    return [X, Y, Z]

@ansor.register_workload_func
def conv1d_nlc(N, L, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, L, CI), name='inputs')
    weight = te.placeholder((kernel_size, CI//groups, CO), name='weight')

    batch_size, in_len, in_channel = inputs.shape
    k_len, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups
    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name='rc')
    rl = te.reduce_axis((0, k_len), name='rl')

    padded = topi.nn.pad(inputs, [0, padding, 0])
    output = te.compute(
        (batch_size, out_len, out_channel),
        lambda n, l, co: te.sum(
            (padded[n, l * stride + rl * dilation, co // out_channel_per_group * channel_per_group + rc] *
             weight[rl, rc, co]), axis=[rl, rc]),
        name='conv1d_nlc'
    )
    return [inputs, weight, output]

@ansor.register_workload_func
def conv2d_nhwc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, H, W, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, CI//groups, CO), name='weight')
    batch_size, in_h, in_w, in_channel = inputs.shape
    k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, co: te.sum(
            (padded[n, h * stride + rh * dilation, w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc]
             * weight[rh, rw, rc, co]), axis=[rh, rw, rc]
        ),
        name='conv2d_nhwc'
    )
    return [inputs, weight, output]

@ansor.register_workload_func
def conv2d_nchw(N, CI, H, W, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, CI, H, W), name='inputs')
    weight = te.placeholder((CO, CI//groups, kernel_size, kernel_size), name='weight')
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w, = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name="rc")
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")

    padded = topi.nn.pad(inputs, [0, 0, padding, padding])
    output = te.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda n, co, h, w: te.sum(
            (padded[n, co // out_channel_per_group * channel_per_group + rc,
                    h * stride + rh * dilation, w * stride + rw * dilation]
             * weight[co, rc, rh, rw]), axis=[rc, rh, rw]
        ),
        name='conv2d_nchw'
    )
    return [inputs, weight, output]

@ansor.register_workload_func
def conv3d_ndhwc(N, D, H, W, CI, CO, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    inputs = te.placeholder((N, D, H, W, CI))
    weight = te.placeholder((kernel_size, kernel_size, kernel_size, CI//groups, CO))
    batch_size, in_d, in_h, in_w, in_channel = inputs.shape
    k_d, k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_d = (in_d + 2 * padding - dilation * (k_d - 1) - 1) // stride + 1
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rd = te.reduce_axis((0, k_d), name='rd')
    rh = te.reduce_axis((0, k_h), name='rh')
    rw = te.reduce_axis((0, k_w), name='rw')
    rc = te.reduce_axis((0, channel_per_group), name='rc')

    padded = topi.nn.pad(inputs, [0, padding, padding, padding, 0])
    output = te.compute(
        (batch_size, out_d, out_h, out_w, out_channel),
        lambda n, d, h, w, co: te.sum(
            (padded[n, d * stride + rd * dilation,
                    h * stride + rh * dilation, w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc]
             * weight[rd, rh, rw, rc, co]),
            axis=[rd, rh, rw, rc]
        ),
        name='conv3d_ndhwc'
    )
    return [inputs, weight, output]

@ansor.register_workload_func
def depthwise_conv2d_nhwc(N, H, W, C, kernel_size, stride=1, padding=0, dilation=1, factor=1):
    inputs = te.placeholder((N, H, W, C))
    weight = te.placeholder((factor, kernel_size, kernel_size, C))

    batch_size, in_h, in_w, in_channel = inputs.shape
    factor, k_h, k_w, in_channel = weight.shape
    out_channel = in_channel * factor

    assert factor.value == 1, "Not optimized for factor != 1"

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name='rh')
    rw = te.reduce_axis((0, k_w), name='rw')

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, c: te.sum(
            (padded[n,  h * stride + rh * dilation, w * stride + rw * dilation, c // factor]
             * weight[c % factor, rh, rw, c // factor]),
            axis=[rh, rw]
        ),
        name="depth_conv2d_nhwc"
    )
    return [inputs, weight, output]

@ansor.register_workload_func
def conv2d_transpose_nhwc(N, H, W, CI, CO, kernel_size, stride=1, padding=0):
    inputs = te.placeholder((N, H, W, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, CI, CO), name='weight')

    batch, in_h, in_w, in_c = inputs.shape
    filter_h, filter_w, in_c, out_c = weight.shape
    stride_h, stride_w = (stride, stride)

    # compute padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = topi.nn.get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    padded = topi.nn.pad(inputs,
                         [0, (bpad_top + stride_h - 1) // stride_h,
                          (bpad_left + stride_w - 1) // stride_w, 0],
                         [0, (bpad_bottom + stride_h - 1) // stride_h,
                          (bpad_right + stride_w - 1) // stride_w, 0])

    # remove extra padding introduced by dilatation
    idxdiv = te.indexdiv
    idxmod = te.indexmod
    border_h = idxmod(stride_h - idxmod(bpad_top, stride_h), stride_h)
    border_w = idxmod(stride_w - idxmod(bpad_left, stride_w), stride_w)

    # dilation stage
    strides = [1, stride_h, stride_w, 1]
    n = len(padded.shape)

    # We should embed this dilation directly into te.compute rather than creating a new te.compute.
    # Only in this way can we use unroll to eliminate the multiplication of zeros.
    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not strides[i] == 1:
                index_tuple.append(idxdiv(indices[i], strides[i]))
                not_zero.append(idxmod(indices[i], strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = te.all(*not_zero)
            return te.if_then_else(not_zero, padded(*index_tuple), tvm.tir.const(0.0, padded.dtype))
        return padded(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    rc = te.reduce_axis((0, in_c), name='rc')
    rh = te.reduce_axis((0, filter_h), name='rh')
    rw = te.reduce_axis((0, filter_w), name='rw')

    output = te.compute(
        (batch, out_h, out_w, out_c),
        lambda n, h, w, co: te.sum(
            _dilate(n, h + rh + border_h, w + rw + border_w, rc) *
            weight[filter_h - 1 - rh, filter_w - 1 - rw, rc, co],
            axis=[rh, rw, rc]),
        name="conv2d_transpose_nhwc",
        attrs={"ansor_always_unroll_inner": ["h", "w", "rh", "rw", "h_c", "w_c"]})
    # todo(lmzheng): add constraints on the tile size of h and w

    return [inputs, weight, output]

@ansor.register_workload_func
def conv2d_capsule_nhwijc(N, H, W, CI, CO, kernel_size, stride=1, padding=0, capsule_size=4):
    inputs = te.placeholder((N, H, W, capsule_size, capsule_size, CI), name='inputs')
    weight = te.placeholder((kernel_size, kernel_size, capsule_size, capsule_size, CI, CO), name='weight')
    batch_size, in_h, in_w, _, _, in_channel = inputs.shape
    k_h, k_w, _, _, _, out_channel = weight.shape

    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    cap_k = te.reduce_axis((0, capsule_size), name='cap_k')
    rc = te.reduce_axis((0, in_channel), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0, 0, 0])
    output = te.compute(
        (batch_size, out_h, out_w, capsule_size, capsule_size, out_channel),
        lambda n, h, w, cap_i, cap_j, co: te.sum(
            (padded[n, h * stride + rh, w * stride + rw, cap_i, cap_k, rc]
             * weight[rh, rw, cap_k, cap_j, rc, co]), axis=[rh, rw, cap_k, rc]
        ),
        name='conv2d_capsule_nhwijc'
    )
    return [inputs, weight, output]


@ansor.register_workload_func
def conv2d_winograd_nhwc(N, H, W, CI, CO, kernel_size=3, stride=1, padding=0, dilation=1):
    # TODO: implement tile_size
    tile_size = 4 #_infer_tile_size(data, kernel)
    inputs = te.placeholder((N, H, W, CI), name='inputs')
    #weight = te.placeholder((kernel_size, kernel_size, CI, CO), name='weight')
    N, H, W, CI = get_const_tuple(inputs.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    # if dilation_h != 1 or dilation_w != 1:
    #     weight = topi.nn.dilate(weight, (1, 1, dilation_h, dilation_w))
    KH = KW = kernel_size
    HPAD, WPAD, _, _ = topi.nn.get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = (stride, stride) if isinstance(stride, int) else stride
    assert HSTR == 1 and WSTR == 1 and KH == KW

    data_pad = topi.nn.pad(inputs, (0, HPAD, WPAD, 0), (0, HPAD, WPAD, 0), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, 'float32')

    H = (H + 2 * HPAD - KH) // HSTR + 1
    W = (W + 2 * WPAD - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW
    r_kh = te.reduce_axis((0, KH), name='r_kh')
    r_kw = te.reduce_axis((0, KW), name='r_kw')
    # kernel_pack = te.compute((alpha, alpha, CO, CI), lambda eps, nu, co, ci:
    #                           weight[0][0][0][0],
    #                           name='kernel_pack')
    kshape = (alpha, alpha, CO, CI)
    kernel_pack = te.placeholder(kshape, inputs.dtype, name="weight")

    idxdiv = te.indexdiv
    idxmod = te.indexmod
    # pack input tile
    input_tile = te.compute((alpha, alpha, P, CI), lambda eps, nu, p, ci:
                             data_pad[idxdiv(p, (nH * nW))][idxmod(idxdiv(p, nW), nH) * m + eps]
                                     [idxmod(p, nW) * m + nu][ci], name='input_tile',)

    # transform data
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    data_pack = te.compute((alpha, alpha, P, CI), lambda eps, nu, p, ci:
                            te.sum(input_tile[r_a][r_b][p][ci] * B[r_a][eps] * B[r_b][nu],
                                    axis=[r_a, r_b]), name='data_pack',
                                    attrs={"ansor_no_split_at_inner": ["eps", "nu", "r_a", "r_b"],
                                           "ansor_last_split_is_one": ["ci", "p"],
                                           "ansor_always_unroll": ["eps", "nu", "r_a", "r_b"],
                                           "ansor_no_cache_write": "True",
                                           })

    # do batch gemm
    ci = te.reduce_axis((0, CI), name='ci')
    bgemm = te.compute((alpha, alpha, P, CO), lambda eps, nu, p, co:
                        te.sum(data_pack[eps][nu][p][ci] *
                                kernel_pack[eps][nu][co][ci],
                                axis=[ci]), name='bgemm')

    # inverse transform
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    inverse = te.compute((m, m, P, CO), lambda vh, vw, p, co:
                          te.sum(bgemm[r_a][r_b][p][co] * A[r_a][vh] * A[r_b][vw],
                                  axis=[r_a, r_b]), name='inverse',
                          attrs={"ansor_no_split_at_inner": ["vh", "vw", "r_a", "r_b"],
                                 "ansor_always_unroll": ["vh", "vw", "r_a", "r_b"],
                                 "ansor_last_split_is_one": ["co", "p"],
                                 "ansor_no_cache_write": "True",
                                 })

    # output
    output = te.compute((N, H, W, CO), lambda n, h, w, co:
                         inverse[idxmod(h, m),
                                 idxmod(w, m),
                                 n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m),
                                 co],
                         name='conv2d_winograd',
                         tag='conv2d_winograd_nhwc',
                         attrs={"ansor_no_split_at_outer": ["n", "h", "w", "co"],})
    return [inputs, kernel_pack, output]

@ansor.register_workload_func
def conv2d_winograd_nchw(N, CI, H, W, CO, kernel_size=3, stride=1, padding=0, dilation=1, precompute=False):
    # TODO: implement tile_size
    tile_size = 4 #_infer_tile_size(data, kernel)
    inputs = te.placeholder((N, CI, H, W), name='inputs')
    #weight = te.placeholder((CO, CI, kernel_size, kernel_size), name='weight')
    N, CI, H, W = get_const_tuple(inputs.shape)
    # if isinstance(dilation, int):
    #     dilation_h = dilation_w = dilation
    # else:
    #     dilation_h, dilation_w = dilation
    # if dilation_h != 1 or dilation_w != 1:
    #     weight = topi.nn.dilate(weight, (1, 1, dilation_h, dilation_w))
    KH = KW = kernel_size
    HPAD, WPAD, _, _ = topi.nn.get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = (stride, stride) if isinstance(stride, int) else stride
    assert HSTR == 1 and WSTR == 1 and KH == KW

    data_pad = topi.nn.pad(inputs, (0, 0, HPAD, WPAD), (0, 0, HPAD, WPAD), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, 'float32')

    H = (H + 2 * HPAD - KH) // HSTR + 1
    W = (W + 2 * WPAD - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW
    r_kh = te.reduce_axis((0, KH), name='r_kh')
    r_kw = te.reduce_axis((0, KW), name='r_kw')
    # kernel_pack = te.compute((alpha, alpha, CI, CO), lambda eps, nu, ci, co:
    #                           weight[0][0][0][0],
    #                           name='kernel_pack')
    kshape = (alpha, alpha, CI, CO)
    kernel_pack = te.placeholder(kshape, inputs.dtype, name="weight")

    idxdiv = te.indexdiv
    idxmod = te.indexmod
    # pack input tile
    input_tile = te.compute((CI, P, alpha, alpha), lambda ci, p, eps, nu:
                             data_pad[idxdiv(p, (nH * nW))][ci][idxmod(idxdiv(p, nW), nH) * m + eps]
                                     [idxmod(p, nW) * m + nu], name='input_tile')

    # transform data
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    data_pack = te.compute((alpha, alpha, CI, P), lambda eps, nu, ci, p:
                            te.sum(input_tile[ci][p][r_a][r_b] * B[r_a][eps] * B[r_b][nu],
                                    axis=[r_a, r_b]), name='data_pack',
                                    attrs={"ansor_no_split_at_inner": ["eps", "nu", "r_a", "r_b"],
                                           "ansor_no_split_at_outer": ["ci", "p"],
                                           "ansor_always_unroll": ["eps", "nu", "r_a", "r_b"],
                                           "ansor_no_cache_write": "True",
                                           })

    # do batch gemm
    ci = te.reduce_axis((0, CI), name='ci')
    bgemm = te.compute((alpha, alpha, CO, P), lambda eps, nu, co, p:
                        te.sum(data_pack[eps][nu][ci][p] *
                                kernel_pack[eps][nu][ci][co],
                                axis=[ci]), name='bgemm')

    # inverse transform
    r_a = te.reduce_axis((0, alpha), 'r_a')
    r_b = te.reduce_axis((0, alpha), 'r_b')
    inverse = te.compute((CO, P, m, m), lambda co, p, vh, vw:
                          te.sum(bgemm[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw],
                                  axis=[r_a, r_b]), name='inverse',
                          attrs={"ansor_no_split_at_outer": ["co", "p", "vh", "vw", "r_a", "r_b"],
                                 "ansor_always_unroll": ["vh", "vw", "r_a", "r_b"],
                                 "ansor_no_cache_write": "True"})

    # output
    output = te.compute((N, CO, H, W), lambda n, co, h, w:
                         inverse[co, n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m),
                                 idxmod(h, m),
                                 idxmod(w, m)],
                         name='conv2d_winograd',
                         attrs={"ansor_no_split_at_outer": ["n", "co", "h", "w"],})
    return [inputs, kernel_pack, output]

# ========================== Subgraphs ==========================

@ansor.register_workload_func
def transpose_batch_matmul(batch, seq_len, n_head, n_dim):
    query = te.placeholder((batch, seq_len, n_head, n_dim), name='query')
    value = te.placeholder((batch, seq_len, n_head, n_dim), name='value')
    query_T = te.compute((batch, n_head, seq_len, n_dim),
                      lambda b, h, l, d: query[b, l, h, d], name="query_T")
    value_T = te.compute((batch, n_head, n_dim, seq_len),
                      lambda b, h, d, l: value[b, l, h, d], name="value_T")
    k = te.reduce_axis((0, n_dim), name='k')
    out = te.compute((batch, n_head, seq_len, seq_len),
                 lambda b, h, i, j: te.sum(query_T[b][h][i][k] * value_T[b][h][k][j], axis=[k]),
                 name='C')
    return [query, value, out]

# ========================== Tune function & Task dicts ==========================

def tune_wkl(task_func_dict, shape_dict, wkl_type, args):
    target = tvm.target.create(args.target)

    for wkl_meta_name, func in task_func_dict.items():
        if not args.wkl in ["all", wkl_type, wkl_meta_name]:
            continue

        log_file = args.log_file or wkl_meta_name + ".json"
        wkl_keys = []
        for shape in shape_dict[wkl_meta_name]:
            if shape[0] == 1:
                shape = list(shape)
                shape[0] = args.batch_size

            wkl_key = ansor.make_workload_key_func(func, shape)
            wkl_keys.append(wkl_key)
            if args.fast_check:
                break

            if not args.tune:
                cost, gflops = replay_workload(
                        wkl_key, target, args.target_host, log_file,
                        args.local_measure, args.rpc_device_key, args.rpc_host,
                        args.rpc_port, args.rpc_num_threads, args.ndk_cc, False)
                # log_line(BenchmarkRecord(target.name, 'gpu' if target.name == 'cuda' else 'cpu', 'subgraph',
                #                          workload_name, "AutoSchedule", "default",
                #                          {"costs": [cost]}, time.time()), args.out_file)

        if args.tune:
            print("========== Tune for %s (%d shapes) ========== " % (wkl_meta_name, len(wkl_keys)))

            load_log_file = args.load_log or log_file
            n_trials = args.n_trials_per_shape * len(wkl_keys)

            tune_option, measure_ctx = create_tune_option(target, log_file,
                    n_trials, args.num_measure_per_iter, args.verbose,
                    args.n_parallel, args.build_timeout, args.local_measure,
                    args.rpc_device_key, args.rpc_host, args.rpc_port,
                    args.rpc_num_threads, args.ndk_cc)

            # tune workloads jointly using JointTuner
            tune_workloads_jointly(wkl_keys, np.ones(len(wkl_keys)), args.task_scheduler,
                                   target, args.target_host, args.policy, args.model_type,
                                   args.load_model, load_log_file, tune_option)

            if measure_ctx:
                del measure_ctx


single_op_task_func_dict = {
    'GMM': batch_matmul_nkkm,
    'C1D': conv1d_nlc,
    'C2D': conv2d_nhwc,
    'C3D': conv3d_ndhwc,
    'GRP': conv2d_nhwc,
    'DIL': conv2d_nhwc,
    'DEP': depthwise_conv2d_nhwc,
    'T2D': conv2d_transpose_nhwc,
    'CAP': conv2d_capsule_nhwijc,
    'NRM': norm_bmn,
    #'SMX': softmax_mn,

#    The following workloads are not in our sinle op evaluation plan.
#    They should be moved to `common.py` and be used by `tune_wkl.py`.
#    'C2D_NCHW': conv2d_nchw,
#    'C2DWG_NHWC': conv2d_winograd_nhwc,
#    'C2DWG_NCHW': conv2d_winograd_nchw,
#    'GMM_TC': matmul_nkkm,
}

subgraph_task_func_dict = {
    'conv2d_bn_relu': conv2d_nhwc_bn_relu,
    #'conv2d_bn_relu': conv2d_nchw_bn_relu,    # some old log uses conv2d_nchw_bn_relu
    'transpose_batch_matmul': transpose_batch_matmul,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Search task related arguments
    parser.add_argument("--wkl", type=str, required=True,
                        help="all      - Tune all workloads; \
                              op       - Tune all single ops; \
                              subgraph - Tune all subgraphs; \
                              specific wkl name - Tune a specific workload")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--fast-check", action='store_true',
                        help='Only run one shape for each workload. This is used for fast checking')

    # Search strategy related arguments
    parser.add_argument("--n-trials-per-shape", type=int, default=1000)
    parser.add_argument("--policy", type=str, choices=['sketch', 'beam-search'], default='sketch')
    parser.add_argument("--model-type", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--task-scheduler", type=str, default='round-robin',
                        choices=['no', 'gradient', 'round-robin'], help='The strategy of task scheduler')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    # Log file related arguments
    parser.add_argument("--log-file", type=str, help="Write measurement records to this log file")
    parser.add_argument("--load-log", type=str, help="Load history log to resume the status of search")
    parser.add_argument("--load-model", type=str, help="Load pre-trained cost model from this file")

    # Measurement related and other arguments
    parser.add_argument("--num-measure-per-iter", type=int, default=48,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--local-measure", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--rpc-device-key", type=str, default=None)
    parser.add_argument("--rpc-host", type=str, default='0.0.0.0')
    parser.add_argument("--rpc-port", type=int, default=9190)
    parser.add_argument("--rpc-num-threads", type=int, default=None)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--ndk-cc", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig()
    logging.getLogger('ansor').setLevel(logging.DEBUG)

    # compute the number of tasks
    num_tasks = 0
    for wkl_meta_name in single_op_task_func_dict:
        if not args.wkl in ["all", "op", wkl_meta_name]:
            continue
        if args.fast_check:
            num_tasks += 1
        else:
            num_tasks += len(single_op_shape_dict[wkl_meta_name])
    for wkl_meta_name in subgraph_task_func_dict:
        if not args.wkl in ["all", "subgraph", wkl_meta_name]:
            continue
        if args.fast_check:
            num_tasks += 1
        else:
            num_tasks += len(subgraph_shape_dict[wkl_meta_name])
    print("Number of tasks: %d\tTotal trials: %d" % (num_tasks, num_tasks * args.n_trials_per_shape))

    # tune for tasks
    tune_wkl(single_op_task_func_dict, single_op_shape_dict, "op", args)
    tune_wkl(subgraph_task_func_dict, subgraph_shape_dict, "subgraph", args)
