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

from . import op as _trn_op


class NKINamespace:
    """The NKI instructions submodule."""

    def __init__(self):
        self.load = _trn_op.nki_load
        self.store = _trn_op.nki_store
        self.tensor_copy = _trn_op.nki_tensor_copy
        self.matmul = _trn_op.nki_matmul
        self.activation = _trn_op.nki_activation
        self.activation_reduce = _trn_op.nki_activation_reduce
        self.reciprocal = _trn_op.nki_reciprocal
        self.tensorreduce = _trn_op.nki_tensorreduce
        self.tensortensor = _trn_op.nki_tensortensor
        self.tensorscalar = _trn_op.nki_tensorscalar
        self.tensorscalar_reduce = _trn_op.nki_tensorscalar_reduce
        self.scalar_tensor_tensor = _trn_op.nki_scalar_tensor_tensor
        self.scalar_tensor_scalar = _trn_op.nki_scalar_tensor_scalar
        self.memset = _trn_op.nki_memset
        self.identity = _trn_op.nki_identity
        self.affine_select = _trn_op.nki_affine_select

    @staticmethod
    def tensorized_instruction():
        """Tensorize the NKI instructions in the region body."""
        from tvm.tirx.script.ir_builder import (
            region,  # pylint: disable=import-outside-toplevel
        )

        return region("tirx.nki.tensorized_instruction", [])


__all__ = ["NKINamespace"]
