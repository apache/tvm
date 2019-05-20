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
# pylint: disable=invalid-name, unused-variable
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
import tvm
import tvm.autotvm as autotvm
from tvm.contrib import cublas
from .tensor_intrin import dp4a
from ..nn.dense import dense, dense_default
from .. import tag
from .. import generic
from ..util import traverse_inline, get_const_tuple


@autotvm.register_topi_compute(dense, ["cuda", "gpu"], "direct")
def dense_cuda(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for cuda backend.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    # pylint: disable=unused-argument
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    target = tvm.target.current_target()
    if "cublas" in target.libs:
        assert out_dtype == data.dtype, "Mixed precision not supported."
        matmul = cublas.matmul(data, weight, False, True)
        if bias is not None:
            matmul = tvm.compute((batch, out_dim), \
                                 lambda i, j: matmul[i, j] + bias[j], \
                                 tag=tag.BROADCAST)
        return matmul
    return dense_default(data, weight, bias, out_dtype)


@autotvm.register_topi_schedule(generic.schedule_dense, ["cuda", "gpu"], "direct")
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    # pylint: disable=unused-argument
    target = tvm.target.current_target()
    if target.target_name == "cuda" and "cublas" in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Dense):
        num_thread = 64
        k = Dense.op.reduce_axis[0]
        ko, kf = s[Dense].split(k, factor=num_thread)
        DenseF = s.rfactor(Dense, kf)

        if Dense.op in s.outputs:
            Out = Dense
        else:
            Out = outs[0].op.output(0)
            s[Dense].compute_at(s[Out], s[Out].op.axis[1])
        s[Out].bind(s[Out].op.axis[0], tvm.thread_axis("blockIdx.y"))
        s[Out].bind(s[Out].op.axis[1], tvm.thread_axis("blockIdx.x"))

        tx = s[Dense].op.reduce_axis[0]
        thread_x = tvm.thread_axis("threadIdx.x")
        s[Dense].bind(tx, thread_x)
        s[DenseF].compute_at(s[Dense], tx)
        s[Dense].set_store_predicate(thread_x.var.equal(0))
        s[Out].set_store_predicate(thread_x.var.equal(0))

    scheduled_ops = []

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            Dense = OP.output(0)
            _schedule(Dense)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


@autotvm.register_topi_compute(dense, ['cuda'], ['int8'])
def dense_int8(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for int8 on CUDA"""
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    k = tvm.reduce_axis((0, in_dim), name='k')

    matmul = tvm.compute((batch, out_dim),
                         lambda i, j: tvm.sum(data[i, k].astype(out_dtype) *
                                              weight[j, k].astype(out_dtype), axis=[k]),
                         tag="dense_int8")

    cfg.add_flop(batch * in_dim * out_dim * 2)

    if bias is not None:
        matmul = tvm.compute((batch, out_dim),
                             lambda i, j: matmul[i, j] + bias[j].astype(out_dtype),
                             tag=tag.BROADCAST)
        cfg.add_flop(batch * out_dim)

    return matmul


@autotvm.register_topi_schedule(generic.schedule_dense, ['cuda', 'gpu'], ['int8'])
def schedule_dense_int8(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])
    def _callback(op):
        if "dense_int8" in op.tag:
            _schedule_dense_int8(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s


_dp4a = dp4a('shared', 'shared', 'local')

def _schedule_dense_int8(cfg, s, output):
    data, weight = s[output].op.input_tensors

    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    in_dim_factor = 4
    assert in_dim % in_dim_factor == 0, "Input dimension must divide {}".format(in_dim_factor)
    if in_dim % 16 == 0:
        in_dim_factor = 16

    # create tuning space
    cfg.define_split("tile_y", batch, num_outputs=4)
    cfg.define_split("tile_x", out_dim, num_outputs=4)
    cfg.define_split("tile_k", in_dim // in_dim_factor, num_outputs=2)
    cfg.define_knob('auto_unroll_max_step', [0, 512, 1500])

    # create cache stage
    AA = s.cache_read(data, 'shared', [output])
    WW = s.cache_read(weight, 'shared', [output])
    CC = s.cache_write(output, 'local')

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    n, x = s[output].op.axis

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    ko = CC.op.reduce_axis[0]
    ko, ki = s[CC].split(ko, factor=4)
    ko, kt = cfg['tile_k'].apply(s, CC, ko)
    s[CC].tensorize(ki, _dp4a)
    by, vy, ty, yi = cfg['tile_y'].apply(s, output, n)
    bx, vx, tx, xi = cfg['tile_x'].apply(s, output, x)

    s[output].reorder(by, bx, vy, vx, ty, tx, yi, xi)
    s[output].bind(by, tvm.thread_axis('blockIdx.y'))
    s[output].bind(bx, tvm.thread_axis('blockIdx.x'))
    s[output].bind(vy, tvm.thread_axis('vthread'))
    s[output].bind(vx, tvm.thread_axis('vthread'))
    s[output].bind(ty, tvm.thread_axis('threadIdx.y'))
    s[output].bind(tx, tvm.thread_axis('threadIdx.x'))
    n_ty = cfg['tile_y'].size[2]
    n_tx = cfg['tile_x'].size[2]

    s[CC].compute_at(s[output], tx)
    yo, xo = CC.op.axis[:2]
    s[CC].reorder(ko, kt, yo, xo, ki)

    for load in [AA, WW]:
        s[load].compute_at(s[CC], ko)

        outer, inner = s[load].split(s[load].op.axis[-1], factor=in_dim_factor)
        s[load].vectorize(inner)
        fused = s[load].op.axis[:-1] + [outer]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        s[load].bind(tx, tvm.thread_axis('threadIdx.x'))
        s[load].bind(ty, tvm.thread_axis('threadIdx.y'))

    s[output].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', False)
    return s
