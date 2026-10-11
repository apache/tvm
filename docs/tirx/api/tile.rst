..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Tensor Instruction Authoring API
================================

Use ``T.cuda.tile`` or ``T.trn.tile`` after ``from tvm.script import tirx as T``.
The signatures below list runtime operands first and static keyword qualifiers
after ``*``. Tensor variables are expanded to full regions. An omitted unary
source defaults to its destination. Named workspace arguments are optional.

See :doc:`../tile_primitives` for memory, scope, descriptor, and synchronization
contracts. These calls produce ordinary IR Calls; the raw scalar instruction
namespaces remain ``T.ptx`` and ``T.nki``.

CUDA
----

.. py:function:: T.cuda.tile.ld(dst, src, *, scope='thread', vec_bits=0, cache=None, l1_evict=None, l2_evict=None, prefetch_size=None)

.. py:function:: T.cuda.tile.st(dst, src, *, scope='thread', vec_bits=0, cache=None, l1_evict=None, l2_evict=None, prefetch_size=None)

.. py:function:: T.cuda.tile.ldmatrix(dst, src, *, scope='thread')

.. py:function:: T.cuda.tile.stmatrix(dst, src, *, scope='thread')

.. py:function:: T.cuda.tile.mov(dst, src, *, scope='thread')

.. py:function:: T.cuda.tile.cp_async(dst, src, predicate=-1, *, scope='thread', direct=False, fill_mode='', prefetch_size=-1)

.. py:function:: T.cuda.tile.cp_async_bulk(dst, src, mbar, remote_cta_id, *, scope='thread')

.. py:function:: T.cuda.tile.cp_async_bulk_tensor_load(dst, src, mbar, cta_mask=0, mbarrier_addr=False, gather4=None, src_selector=None, cache_policy=None, *, scope='thread', descriptor_mode='auto', cta_group=1, cache_hint='', tma_dtype=None, oob=None, prefetch_tensormap=False, tensormap_l2_promotion=None, reduce_op=None)

.. py:function:: T.cuda.tile.cp_async_bulk_tensor_store(dst, src, cache_policy=None, *, scope='thread', descriptor_mode='auto', cta_group=1, cache_hint='', tma_dtype=None, oob=None, prefetch_tensormap=False, tensormap_l2_promotion=None, reduce_op=None)

.. py:function:: T.cuda.tile.cp_reduce_async_bulk_tensor(dst, src, cache_policy=None, *, scope='thread', descriptor_mode='auto', cta_group=1, cache_hint='', tma_dtype=None, oob=None, prefetch_tensormap=False, tensormap_l2_promotion=None, reduce_op=None)

.. py:function:: T.cuda.tile.tcgen05.cp(dst, src, *, scope='thread', shape=None, multicast=None, cta_group=1, decompress=None)

.. py:function:: T.cuda.tile.tcgen05.ld(dst, src, *, scope='thread')

.. py:function:: T.cuda.tile.tcgen05.st(dst, src, *, scope='thread')

.. py:function:: T.cuda.tile.mma_sync(D, A, B, C, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0, *, scope='thread')

.. py:function:: T.cuda.tile.tcgen05.mma(C, A, B, transA=False, transB=False, accum=False, descI=None, *, scope='thread', cta_group=1, mma_m=None, mma_n=None, smem_desc='hoist', is_AB_tf32=False, weight_stationary=None)

.. py:function:: T.cuda.tile.tcgen05.mma_block_scale(C, A, B, SFA, SFB, transA=False, transB=False, accum=False, descI=None, *, scope='thread', cta_group=1, mma_m=None, mma_n=None, smem_desc='hoist', is_AB_tf32=False, weight_stationary=None)

.. py:function:: T.cuda.tile.add(dst, lhs, rhs, *, scope='thread', rounding_mode=None)

.. py:function:: T.cuda.tile.sub(dst, lhs, rhs, *, scope='thread', rounding_mode=None)

.. py:function:: T.cuda.tile.mul(dst, lhs, rhs, *, scope='thread', rounding_mode=None)

.. py:function:: T.cuda.tile.div(dst, lhs, rhs, *, scope='thread', rounding_mode=None)

.. py:function:: T.cuda.tile.max(dst, lhs, rhs, *, scope='thread', rounding_mode=None)

.. py:function:: T.cuda.tile.cvt(dst, src=None, *, scope='thread')

.. py:function:: T.cuda.tile.sqrt(dst, src=None, *, scope='thread')

.. py:function:: T.cuda.tile.ex2(dst, src=None, *, scope='thread')

.. py:function:: T.cuda.tile.lg2(dst, src=None, *, scope='thread')

.. py:function:: T.cuda.tile.compose.exp(dst, src=None, *, scope='thread')

.. py:function:: T.cuda.tile.compose.silu(dst, src=None, *, scope='thread')

.. py:function:: T.cuda.tile.fma(dst, src, scale, bias, *, scope='thread')

.. py:function:: T.cuda.tile.compose.exp_with_scale_bias(dst, src, scale, bias, *, scope='thread')

.. py:function:: T.cuda.tile.compose.exp2_with_scale_bias(dst, src, scale, bias, *, scope='thread')

.. py:function:: T.cuda.tile.compose.log2_with_scale_bias(dst, src, scale, bias, *, scope='thread')

.. py:function:: T.cuda.tile.compose.sqrt_with_scale_bias(dst, src, scale, bias, *, scope='thread')

TRN
---

.. py:function:: T.trn.tile.load(dst, src, *, scope='thread')

.. py:function:: T.trn.tile.store(dst, src, *, scope='thread')

.. py:function:: T.trn.tile.tensor_copy(dst, src, *, scope='thread', max_inst_size=512)

.. py:function:: T.trn.tile.matmul(D, A, B, C=None, transpose_A=False, transpose_B=False, alpha=1.0, beta=0.0, acc_psum=None, *, scope='thread')

.. py:function:: T.trn.tile.reciprocal(dst, src, *, scope='thread', max_inst_size=512)

.. py:function:: T.trn.tile.memset(dst, src, *, scope='thread', max_inst_size=512)

.. py:function:: T.trn.tile.activation(dst, src, scale=1.0, bias=0.0, const_bias=None, *, scope='thread', opcode='exp', max_inst_size=512)

.. py:function:: T.trn.tile.tensortensor(dst, lhs, rhs, *, scope='thread', opcode='add', max_inst_size=512)

.. py:function:: T.trn.tile.tensorscalar(dst, lhs, rhs, *, scope='thread', opcode='add', max_inst_size=512)

.. py:function:: T.trn.tile.tensorreduce(dst, src, partial_reduce=None, *, scope='thread', reduce_op='sum', axes=[-1], negate=False, max_inst_size=None)

.. py:function:: T.trn.tile.scalar_tensor_scalar(dst, data, operand0, operand1, *, scope='thread', op0='mul', op1='add', reverse1=False, max_inst_size=512)

.. py:function:: T.trn.tile.scalar_tensor_tensor(dst, data, operand0, operand1, *, scope='thread', op0='mul', op1='add', reverse1=False, max_inst_size=512)

.. py:function:: T.trn.tile.tensorscalar_reduce(dst, reduced, lhs, rhs, partial_reduce=None, *, scope='thread', opcode='add', reduce_op='sum', axes=[-1], max_inst_size=None)

.. py:function:: T.trn.tile.activation_reduce(dst, reduced, src, scale=1.0, bias=0.0, const_bias=None, partial_reduce=None, *, scope='thread', opcode='exp', reduce_op='sum', axes=[-1], max_inst_size=None)

.. py:function:: T.trn.tile.affine_select(dst, true_value, false_value, pred, *, scope='thread')
