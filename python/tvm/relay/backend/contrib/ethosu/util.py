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
"""
Helper utility Enums and Functions used through out code generation.

The rest of the utility functions are misc.
Refer to the description inside such functions
"""

from inspect import signature
from enum import Enum
from typing import Union, Tuple, List
import numpy as np  # type: ignore

import tvm  # type: ignore
from tvm import relay
from tvm._ffi import register_object
from tvm.runtime import Object
from . import _ffi_api


class QConv2DArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.conv2d arguments.
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class QConv2DTransposeArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.conv2d_transpose arguments.
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class RequantArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.requantize arguments.
    """

    IFM_SCALE = 1
    IFM_ZERO_POINT = 2
    OFM_SCALE = 3
    OFM_ZERO_POINT = 4


class BiasAddArgs(Enum):
    """
    This is a helper enums to obtain the correct index
    of qnn.bias_add arguments.
    """

    BIASES = 1


class ClipArgs(Enum):
    """
    This is a helper enums to obtain the correct index
    of clip arguments.
    """

    A_MIN = 1
    A_MAX = 2


class BinaryElementwiseArgs(Enum):
    """This is a helper enums to access the correct index
    of binary elementwise arguments
    """

    IFM = 0
    IFM2 = 1
    IFM_SCALE = 2
    IFM_ZERO_POINT = 3
    IFM2_SCALE = 4
    IFM2_ZERO_POINT = 5
    OFM_SCALE = 6
    OFM_ZERO_POINT = 7


class QuantizeArgs(Enum):
    """
    This is a helper enums to access the correct index of
    quantize arguments
    """

    IFM = 0
    OFM_SCALE = 1
    OFM_ZERO_POINT = 2


class DequantizeArgs(Enum):
    """
    This is a helper enums to access the correct index of
    dequantize arguments
    """

    IFM = 0
    IFM_SCALE = 1
    IFM_ZERO_POINT = 2


class QDenseArgs(Enum):
    """
    This is a helper enum to access the correct index of
    qnn.dense arguments
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


def is_npu_func(func: relay.Function) -> bool:
    """Check if the given function is an NPU function."""
    return func.attrs and "Compiler" in func.attrs and func.attrs["Compiler"] == "ethos-u"


def is_composite_func(func: relay.Function, name: str) -> bool:
    """
    This method checks whether the call is to
    a composite function of a given name.

    Parameters
    ----------
    func : relay.Function
        The header to be displayed along with the dump.

    name : str
        The candidate name to be checked

    Returns
    --------
    a boolean
    """

    if not hasattr(func, "attrs"):
        return False
    if "Composite" not in func.attrs.keys():
        return False
    composite_name = func.attrs["Composite"]

    return composite_name == name


def is_named_ethosu_op(expr: tvm.relay.Expr, name: str) -> bool:
    """Checks whether a relay expression matches that of the
    named operator.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The expression to check.
    name : str
        The name of the expected operator
        (without NPU prefix "contrib.ethosu").

    Returns
    -------
    bool
        True if expression matches name, false if not.
    """
    prefix = "contrib.ethosu."
    return (
        isinstance(expr, tvm.relay.expr.Call)
        and isinstance(expr.op, tvm.ir.op.Op)
        and expr.op.name == prefix + name
    )


def get_range_for_dtype_str(dtype: str) -> Tuple[int, int]:
    """
    Produce the min,max for a give data type.

    Parameters
    ----------
    dtype : str
        a type string (e.g., int8)

    Returns
    -------
    type_info.min : int
        the minimum of the range
    type_info.max : int
        the maximum of the range
    """

    try:
        type_info = np.iinfo(dtype)
    except ValueError:
        type_info = np.finfo(dtype)
    return type_info.min, type_info.max


def round_away_zero(f: Union[float, np.double, np.single, np.float32, np.float64]) -> np.float64:
    """Round the number away from zero towards +inf / -inf"""
    offset = -0.5 if (f < 0) else 0.5
    return np.trunc(f + offset)


def round_up(a: int, b: int) -> int:
    """Round up to a multiple of b"""
    return ((a + b - 1) // b) * b


def get_accelerator_config():
    """Get the variant of the accelerator to compile for"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
    return compiler_attrs.accelerator_config


def is_cascader_enabled():
    """Determine whether the cascader is enabled"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
    return compiler_attrs.enable_cascader


def is_striping_enabled():
    """Determine whether the cascader is enabled"""
    compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
    return compiler_attrs.enable_striping


def get_arg_count(func):
    """Helper function to get the number of
    arguments in a python function"""
    sig = signature(func)
    return len(sig.parameters)


def get_dim_value(layout: str, dim: int):
    """This is a helper function to retrieve the value
    of the dimension given the shape and the layout
    """
    assert isinstance(layout, str)
    assert dim in list(layout)
    for idx, dim_char in enumerate(layout):
        if dim_char == dim:
            return idx
    return None


def calculate_size_bytes(expr):
    """This is a helper function to calculate the number
    of bytes required to hold the tensor/relay.expr"""
    try:
        type_info = np.iinfo(expr.checked_type.dtype)
    except ValueError:
        type_info = np.finfo(expr.checked_type.dtype)
    element_size = type_info.bits // 8
    elements = np.prod(list(expr.checked_type.shape))
    return element_size * elements


@register_object("relay.ext.ethos-u.BaseAddress")
class BaseAddress(Object):
    """
    This is a structure to hold base addresses for pointers
    provided for the driver.
    """

    def __init__(
        self,
        name: str,
        primfunc_param_idx: int,
        region: int,
        size: int,
        is_runtime_allocation: bool = False,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.BaseAddress,  # type: ignore # pylint: disable=no-member
            name,
            primfunc_param_idx,
            region,
            size,
            is_runtime_allocation,
        )


@register_object("relay.ext.ethos-u.CompilationArtifact")
class CompilationArtifact(Object):
    """
    This is a structure to hold binary artifacts
    for the microNPU.
    """

    def __init__(
        self,
        function_name: str,
        command_stream: str,
        encoded_constants: str,
        base_addresses: List[BaseAddress],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.CompilationArtifact,  # type: ignore # pylint: disable=no-member
            function_name,
            command_stream,
            encoded_constants,
            base_addresses,
        )


def create_npu_function_pass(opt_level: int, name: str = ""):
    """
    A utility decorator that wraps a given class as an NPU function pass. That is,
    a pass that behaves like a function pass and only traverses NPU external
    functions. How each NPU function is mutated is defined by the
    `transform_npu_function(global_variable, relay_function)` function which should
    be created in the class that is to be decorated. See the example below.

    Example
    -------
    This small example demonstrates a pass over NPU functions that performs no
    mutation.

    @create_npu_function_pass(opt_level=1)
    class MyPass:
        def transform_npu_function(self, global_var, func):
            return func

    mod = tvm.IRModule()
    mod = MyPass()(mod)

    Parameters
    ----------
    opt_level: int
        Optimization level for the module pass.
    name: str, optional
        Name for the module pass.

    Returns
    -------
    decorator
        The npu_pass decorator.
    """

    def decorator(npu_pass_class):
        @tvm.ir.transform.module_pass(name=name, opt_level=opt_level)
        class ModulePassWrapper:
            """The wrapper for the NPU pass."""

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def transform_module(self, mod: tvm.ir.IRModule, _) -> tvm.ir.IRModule:
                npu_functions = filter(lambda x: is_npu_func(x[1]), mod.functions.items())
                for global_var, func in npu_functions:
                    npu_pass = npu_pass_class(*self.args, **self.kwargs)
                    func = npu_pass.transform_npu_function(global_var, func)
                    mod.update_func(global_var, func)
                return mod

        return ModulePassWrapper

    return decorator
