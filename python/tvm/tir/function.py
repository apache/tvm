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
"""Function data types."""

from typing import Mapping, Union

import tvm._ffi
import tvm.runtime
from tvm.runtime import Object
from tvm.ir import BaseFunc
from .buffer import Buffer
from .expr import Var, PrimExpr
from . import _ffi_api


@tvm._ffi.register_object("tir.PrimFunc")
class PrimFunc(BaseFunc):
    """A function declaration expression.

    Parameters
    ----------
    params: List[Union[tvm.tir.Var, tvm.tir.Buffer]]
        List of input parameters to the function.

    body: tvm.tir.Stmt
        The body of the function.

    ret_type: tvm.ir.Type
        The return type annotation of the function.

    buffer_map : Map[tvm.tir.Var, tvm.tir.Buffer]
        The buffer binding map.

    attrs: Optional[tvm.Attrs]
        Attributes of the function, can be None

    resource_handle: Optional[tvm.tir.Var]
        The resource handle to be used by the function when accessing platform resources,
        if not passed a Var will be created for it

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(
        self,
        params,
        body,
        ret_type=None,
        buffer_map=None,
        attrs=None,
        resource_handle=None,
        span=None,
    ):
        param_list = []
        buffer_map = {} if buffer_map is None else buffer_map

        # This is bound later as it relies on the FFI API having defined "Var"
        if resource_handle is None:
            resource_handle = Var("resource_handle", dtype="handle")

        for x in params:
            x = tvm.runtime.convert(x) if not isinstance(x, Object) else x
            if isinstance(x, Buffer):
                var = Var(x.name, dtype="handle")
                param_list.append(var)
                buffer_map[var] = x
            elif isinstance(x, Var):
                param_list.append(x)
            else:
                raise TypeError("params can only contain Var or Buffer")

        self.__init_handle_by_constructor__(
            _ffi_api.PrimFunc, param_list, body, ret_type, buffer_map, attrs, resource_handle, span  # type: ignore
        )

    def with_body(self, new_body, span=None):
        """Create a new PrimFunc with the same set signatures but a new body.

        Parameters
        ----------
        new_body : Stmt
            The new body.

        span : Optional[Span]
            The location of this itervar in the source code.

        Returns
        -------
        new_func : PrimFunc
            The created new function.
        """
        return PrimFunc(self.params, new_body, self.ret_type, self.buffer_map, self.attrs, span)

    def specialize(self, param_map: Mapping[Var, Union[PrimExpr, Buffer]]):
        """Specialize parameters of PrimFunc

        Parameters
        ----------

        param_map : Mapping[Var, Union[PrimExpr, Buffer]]
            The mapping from function params to the instance

        Examples
        --------
        We can define a Meta TIR function with symbolic shape:

        .. code-block:: python

            @tvm.script.tir
            def mem_copy(a: ty.handle, b: ty.handle, m: ty.int32, n: ty.int32) -> None:
                A = tir.match_buffer(a, (m, n), "float32")
                B = tir.match_buffer(b, (m, n), "float32")

                with tir.block([m, n], "") as [vi, vj]:
                    B[vi, vj] = A[vi, vj]

        Then we can make it specialized with given shapes or buffers.

        .. code-block:: python

            a, _, m, n = mem_copy.params
            func = mem_copy.specialize({a: tir.decl_buffer((16, 16))})
            # or
            func = mem_copy.specialize({n: 16, m: 16})

        The specialized function:

        .. code-block:: python

            @tvm.script.tir
            def mem_copy_16_16(a: ty.handle, b: ty.handle) -> None:
                A = tir.match_buffer(a, (16, 16), "float32")
                B = tir.match_buffer(b, (16, 16), "float32")

                with tir.block([16, 16], "") as [vi, vj]:
                    B[vi, vj] = A[vi, vj]

        Returns
        -------
        func : PrimFunc
            The new function with parameter specialized
        """
        return _ffi_api.Specialize(self, param_map)  # type: ignore
