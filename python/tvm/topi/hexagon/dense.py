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
# pylint: disable=invalid-name
"""Schedule for dense operator"""

import tvm
from tvm.topi.utils import traverse_inline
from tvm import te
from .. import tag
from .tensor_intrin import dot_vrmpy


def schedule_dense(outs):
    """Schedule for dense op.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense in the format
        of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = tvm.te.create_schedule([x.op for x in outs])
    tvm.te.schedule.AutoInlineInjective(s)
    return s


def dense_u8u8i32_vrmpy_compute(X, packed_w, bias, out_dtype):
    """Compute for uint8 x uint8 -> int32 dense using vrmpy"""
    assert X.dtype == "uint8" and packed_w.dtype == "uint8" and out_dtype == "int32"
    m, k = X.shape
    n_o, _, n_i, _ = packed_w.shape
    assert n_i == 32
    ak = te.reduce_axis((0, k), name="k")

    C = te.compute(
        (m, n_o * n_i),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packed_w[tvm.tir.indexdiv(j, 32), tvm.tir.indexdiv(ak, 4), j % 32, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        tag="dense_u8u8i32_vrmpy",
        name="compute",
    )

    if bias is not None:
        C = te.compute(C.shape, lambda i, j: C[i, j] + bias[j], tag=tag.BROADCAST)

    return C


def dense_u8u8i32_vrmpy_schedule(outs):
    """Schedule for vrmpy dense"""
    s = te.create_schedule([x.op for x in outs])
    # O: The output of the fused op
    O = outs[0]

    def _schedule_dense(s, C, O):
        (a_k,) = C.op.reduce_axis
        a_y = C.op.axis[-2]
        a_yo, a_yi = s[C].split(a_y, factor=32)
        a_xo, a_xi = s[C].split(C.op.axis[-1], factor=32)
        a_ko, a_ki = s[C].split(a_k, factor=4)

        s[C].reorder(a_yo, a_xo, a_yi, a_ko, a_xi, a_ki)

        pc = dot_vrmpy("uint8", "uint8")
        s[C].tensorize(a_xi, pc)
        s[C].parallel(s[C].fuse(a_yo, a_xo))

        if C != O:
            a_y = O.op.axis[-2]
            a_yo, a_yi = s[O].split(a_y, factor=32)
            a_xo, a_xi = s[O].split(O.op.axis[-1], factor=32)

            s[O].reorder(a_yo, a_xo, a_yi, a_xi)
            s[O].vectorize(a_xi)
            s[C].compute_at(s[O], a_yi)
            s[O].parallel(s[O].fuse(a_yo, a_xo))

    def _callback(op):
        if "u8u8i32_vrmpy" in op.tag:
            # C: The output of GEMM
            C = op.output(0)
            _schedule_dense(s, C, O)

    traverse_inline(s, outs[0].op, _callback)

    return s
