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
"""Type relation and function for type checking."""
import tvm._ffi

from . import _ffi_api
from .type import Type


@tvm._ffi.register_object("relay.TensorType")
class TensorType(Type):
    """A concrete TensorType in Relay.

    This is the type assigned to tensors with a known dtype and shape.
    For example, a tensor of `float32` and `(5, 5)`.

    Parameters
    ----------
    shape : List[tvm.ir.PrimExpr]
        The shape of the Tensor

    dtype : Optional[str]
        The content data type.
    """

    def __init__(self, shape, dtype="float32"):
        self.__init_handle_by_constructor__(_ffi_api.TensorType, shape, dtype)

    @property
    def concrete_shape(self):
        """Get shape of the type as concrete tuple of int.

        Returns
        -------
        shape : List[int]
            The concrete shape of the Type.

        Raises
        ------
        TypeError : If the shape is symbolic
        """
        return tuple(int(x) for x in self.shape)

    def __str__(self):
        from tvm.relay import pretty_print  # pylint: disable=import-outside-toplevel

        return pretty_print(self)
