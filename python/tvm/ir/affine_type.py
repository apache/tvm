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
"""Types for quantized Tensors."""
import tvm._ffi

from . import _ffi_api
from .base import Node


class AffineType(Node):
    """The base class of Affine Types."""

    def __eq__(self, other):
        """Compare two types for structural equivalence."""
        return bool(tvm.ir.structural_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        from tvm.relay import pretty_print  # pylint: disable=import-outside-toplevel

        return pretty_print(self)


@tvm._ffi.register_object("TensorAffineType")
class TensorAffineType(AffineType):
    """The quantized type of a tensor, with scale, zero point, and datatype

    The real space value is calculated as x = x_q * scale + zero_point

    Parameters
    ----------
    scale: Expr
        The scale

    zero_point: Expr
        The zero_point

    dtype : str
        The content data type.

    axis : int
        The axis for per-channel quantization.
    """

    def __init__(self, scale, zero_point, dtype, axis=-1):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorAffineType, scale, zero_point, dtype, axis
        )


@tvm._ffi.register_object("TupleAffineType")
class TupleAffineType(AffineType):
    """Affine types of a node with multiple outputs

    Parameters
    ----------
    types : List[TensorAffineType]
        The shape of the Tensor

    """

    def __init__(self, types):
        self.__init_handle_by_constructor__(_ffi_api.TupleAffineType, types)
