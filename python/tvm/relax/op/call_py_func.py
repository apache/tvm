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
"""Relax call_py_func operator."""

from typing import List, Optional, Union

from tvm import relax
from tvm.ir import Op
from tvm.relax import Call, Expr, Var
from tvm.relax.expr import Call as RelaxCall
from tvm.relax.struct_info import StructInfo


def call_py_func(
    func_name: str,
    args: List[Expr],
    struct_info: Optional[StructInfo] = None,
) -> RelaxCall:
    """Call a Python function from Relax.
    
    This operator allows Relax functions to invoke Python functions
    that are stored in the IRModule's pyfuncs attribute.
    
    Parameters
    ----------
    func_name : str
        The name of the Python function to call.
        
    args : List[Expr]
        The arguments to pass to the Python function.
        
    struct_info : Optional[StructInfo]
        The expected return type of the function call.
        If not provided, it will be inferred.
        
    Returns
    -------
    RelaxCall
        A call expression that will invoke the Python function at runtime.
    """
    # For now, we'll create a simple call that can be recognized by our printer
    # We'll use a custom operator name that our system can handle
    
    # Create a simple call with a custom operator name
    from tvm.relax import Call, PrimValue, StringImm
    from tvm.relax import TensorStructInfo, ObjectStructInfo
    
    # Create a custom call that our printer can recognize
    # We'll use a string literal to encode the function name
    func_name_expr = StringImm(func_name)
    
    # Create a tuple of arguments
    from tvm.relax import Tuple
    args_tuple = Tuple(args)
    
    # Create a simple call structure that our printer can handle
    # We'll use a custom format: call_py_func_internal(func_name, args)
    from tvm.relax import Var
    from tvm.relax.struct_info import FuncStructInfo, ObjectStructInfo
    
    # Create a dummy function with the right signature
    dummy_func = Var("__call_py_func_internal__", 
                     FuncStructInfo([ObjectStructInfo(), ObjectStructInfo()], ObjectStructInfo()))
    
    # Create the call
    call = Call(dummy_func, [func_name_expr, args_tuple])
    
    # Set the struct info if provided
    if struct_info is not None:
        call.struct_info_ = struct_info
    
    return call


def _infer_struct_info_call_py_func(call: RelaxCall, ctx) -> StructInfo:
    """Infer the struct info for call_py_func calls.
    
    Since Python functions can return any type, we use a conservative
    approach and return ObjectStructInfo() unless explicitly specified.
    """
    # If struct info is already set, use it
    if call.struct_info_ is not None:
        return call.struct_info_
    
    # Otherwise, return ObjectStructInfo as a safe default
    return relax.ObjectStructInfo()


# Note: The actual operator registration happens in C++ code
# This Python file provides the Python interface for call_py_func
