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

import tvm._ffi
import tvm.runtime
from tvm.runtime import Object
from tvm.ir import BaseFunc
from .buffer import Buffer
from .expr import Var
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
    """
    def __init__(self,
                 params,
                 body,
                 ret_type=None,
                 buffer_map=None,
                 attrs=None):
        param_list = []
        buffer_map = {} if buffer_map is None else buffer_map
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
            _ffi_api.PrimFunc, param_list, body, ret_type, buffer_map, attrs)

    def with_attr(self, attr_key, attr_value):
        """Create a new copy of the function and update the attribute

        Parameters
        ----------
        attr_key : str
            The attribute key to use.

        attr_value : Object
            The new attribute value.

        Returns
        -------
        func : Function
            A new copy of the function
        """
        return _ffi_api.PrimFuncWithAttr(
            self, attr_key, tvm.runtime.convert(attr_value))
