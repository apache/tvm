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
# pylint: disable=unrecognized-inline-option
"""Function data types."""

from collections.abc import Mapping, Sequence

import tvm_ffi

import tvm_tirx
import tvm_tirx.runtime
from tvm_tirx.ir import BaseFunc, Scriptable, SeqStmt, Stmt, Var
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api
from .expr import Expr


@tvm_ffi.register_object("tirx.Function")
class Function(BaseFunc, Scriptable):
    """A function declaration expression.

    Parameters
    ----------
    params: List[Union[tvm_tirx.tirx.Var, tvm_tirx.tirx.Var]]
        List of input parameters to the function.

    body: Stmt | Sequence[Stmt] | None
        The body of the function, or None for a declaration.

    ret_type: tvm_tirx.ir.Type
        The return type annotation of the function.

    attrs: Optional[tvm.Attrs]
        Attributes of the function, can be None

    loc : Location, optional
        The location of this itervar in the source code.
    """

    body: SeqStmt | None

    def __init__(
        self,
        params,
        body: Stmt | Sequence[Stmt] | None,
        ret_type=None,
        attrs=None,
        loc: Location = UNKNOWN_LOC,
    ):
        if ret_type is None:
            ret_type = tvm_tirx.ir.Type.missing()
        param_list = []
        for x in params:
            x = tvm_ffi.convert(x) if not isinstance(x, tvm_ffi.Object) else x
            if not isinstance(x, Var):
                raise TypeError("params can only contain Var or Var")
            param_list.append(x)

        if attrs is None:
            attrs = tvm_tirx.ir.make_node("ir.DictAttrs")

        self.__init_handle_by_constructor__(
            _ffi_api.Function,
            param_list,
            body,
            ret_type,
            attrs,
            loc,
        )  # type: ignore

    def with_body(self, new_body: Stmt | Sequence[Stmt] | None, loc: Location = UNKNOWN_LOC):
        """Create a new Function with the same set signatures but a new body.

        Parameters
        ----------
        new_body : Stmt | Sequence[Stmt] | None
            The new body, or None for a declaration.

        loc : Location, optional
            The location of this itervar in the source code.

        Returns
        -------
        new_func : Function
            The created new function.
        """
        return Function(
            self.params,
            new_body,
            ret_type=self.ret_type,
            attrs=self.attrs,
            loc=loc,
        )

    def specialize(self, param_map: Mapping[Var, Expr | Var]):
        """Specialize parameters of Function

        Parameters
        ----------

        param_map : Mapping[Var, Union[Expr, Var]]
            The mapping from function params to the instance

        Examples
        --------
        We can define a TIRX function with symbolic shape:

        .. code-block:: python

            from __future__ import annotations

            @T.function
            def mem_copy(
                A: T.Tensor((m, n), "float32"),
                B: T.Tensor((m, n), "float32"),
                m: T.int32,
                n: T.int32,
            ) -> None:

                for i, j in T.grid(m, n):
                    B[i, j] = A[i, j]

        Then we can make it specialized with given shapes or tensors.

        .. code-block:: python

            a, _, m, n = mem_copy.params
            func = mem_copy.specialize({a: tirx.decl_tensor((16, 16))})
            # or
            func = mem_copy.specialize({n: 16, m: 16})

        The specialized function:

        .. code-block:: python

            @T.function
            def mem_copy_16_16(
                A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")
            ) -> None:

                for i, j in T.grid(16, 16):
                    B[i, j] = A[i, j]

        Returns
        -------
        func : Function
            The new function with parameter specialized
        """
        return _ffi_api.Specialize(self, param_map)  # type: ignore


def renew_def(func: Function) -> Function:
    """Renew variable definitions and remap their uses in a function.

    Definition regions follow the registered structural hooks. Free variables
    and function attributes are preserved, and the function signature is rebuilt.

    Parameters
    ----------
    func : Function
        The function to renew.

    Returns
    -------
    Function
        The function with fresh variable definitions.
    """
    return _ffi_api.RenewDef(func)
