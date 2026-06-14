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
"""Trainium TVMScript namespaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import op as _trn_op

OpWrapper = Callable[[Callable[..., Any]], Callable[..., Any]]


def _default_op_wrapper() -> OpWrapper:
    from tvm.tirx.script.builder.ir import _op_wrapper  # pylint: disable=import-outside-toplevel

    return _op_wrapper


class NKINamespace:
    """The NKI instructions submodule."""

    def __init__(self, op_wrapper: OpWrapper | None = None):
        wrap = op_wrapper or _default_op_wrapper()
        self.load = wrap(_trn_op.nki_load)
        self.store = wrap(_trn_op.nki_store)
        self.tensor_copy = wrap(_trn_op.nki_tensor_copy)
        self.matmul = wrap(_trn_op.nki_matmul)
        self.activation = wrap(_trn_op.nki_activation)
        self.activation_reduce = wrap(_trn_op.nki_activation_reduce)
        self.reciprocal = wrap(_trn_op.nki_reciprocal)
        self.tensorreduce = wrap(_trn_op.nki_tensorreduce)
        self.tensortensor = wrap(_trn_op.nki_tensortensor)
        self.tensorscalar = wrap(_trn_op.nki_tensorscalar)
        self.tensorscalar_reduce = wrap(_trn_op.nki_tensorscalar_reduce)
        self.scalar_tensor_tensor = wrap(_trn_op.nki_scalar_tensor_tensor)
        self.scalar_tensor_scalar = wrap(_trn_op.nki_scalar_tensor_scalar)
        self.memset = wrap(_trn_op.nki_memset)
        self.identity = wrap(_trn_op.nki_identity)
        self.affine_select = wrap(_trn_op.nki_affine_select)


__all__ = ["NKINamespace", "OpWrapper"]
