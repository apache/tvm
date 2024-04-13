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
# pylint: disable=unused-argument
"""tvm.contrib.msc.framework.tensorrt.transform.pattern"""

from typing import Mapping, Tuple, List, Union, Callable, Dict
from functools import wraps, partial

import tvm
from tvm import relax
from tvm.relax.dpl import pattern
from tvm.relax.transform import PatternCheckContext, FusionPattern
from tvm.relax.backend.pattern_registry import register_patterns
from tvm.contrib.msc.core.transform import pattern as msc_pattern
from tvm.contrib.msc.core import _ffi_api


def basic_pattern(
    op_name: str, input_types: List[str] = None
) -> Tuple[pattern.DFPattern, Mapping[str, pattern.DFPattern]]:
    """create basic pattern for tensorrt support ops.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.nn.conv2d"
    input_types: list<str>
        The input types, elach element can be input| constant

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing the operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    input_types = input_types or ["input"]
    inputs = []
    for i_type in input_types:
        if i_type == "input":
            inputs.append(pattern.wildcard())
        elif i_type == "constant":
            inputs.append(pattern.is_const())
        else:
            raise Exception("Unexpected input type " + str(i_type))
    out = pattern.is_op(op_name)(*inputs)
    annotations = {"input_" + str(idx): arg for idx, arg in enumerate(inputs)}
    annotations["out"] = out
    return out, annotations


def elemwise_pattern(op_name: str) -> Tuple[pattern.DFPattern, Mapping[str, pattern.DFPattern]]:
    """create elemwise pattern for tensorrt support ops.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.add"

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing the operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    return basic_pattern(op_name, ["input", "input"])


def argmaxmin_pattern(op_name: str) -> Tuple[pattern.DFPattern, Mapping[str, pattern.DFPattern]]:
    """create argmaxmin pattern for tensorrt support ops.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.argmax"

    Returns
    -------
    out: tvm.relax.dpl.pattern.DFPattern
        The resulting pattern describing the operation.

    annotations: Mapping[str, tvm.relax.dpl.pattern.DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    data = pattern.wildcard()
    argmaxmin = pattern.is_op(op_name)(data)
    out = pattern.is_op("relax.astype")(argmaxmin)
    return out, {"input": data, "argmaxmin": argmaxmin, "out": out}


def _check_expr(expr: relax.Expr, dtypes: Tuple[str] = None) -> bool:
    """Check if the expr can be fused on tensorrt.

    Parameters
    ----------
    expr: relax.Expr
        The expr to be check
    dtype: tuple<str>
        The accept dtypes

    Returns
    -------
    pass: bool
        Whether the expr is correct.
    """

    if isinstance(expr, relax.ShapeExpr):
        return True
    if isinstance(expr, relax.PrimValue):
        return True
    if isinstance(expr, relax.Tuple):
        return all(_check_expr(field) for field in expr.fields)
    if any(i < 0 for i in expr.struct_info.shape.values):
        return False
    dtypes = dtypes or ("float32", "float16")
    if expr.struct_info.dtype not in dtypes:
        return False
    return True


def _basic_check(context: PatternCheckContext) -> bool:
    """Check if the basic pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    for _, expr in context.annotated_expr.items():
        if not _check_expr(expr):
            return False
    return True


def _argmaxmin_check(context: PatternCheckContext) -> bool:
    """Check if the argmaxmin pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if not _check_expr(context.annotated_expr["input"]):
        return False
    return _check_expr(context.annotated_expr["out"], ("int32"))


def _compare_check(context: PatternCheckContext) -> bool:
    """Check if the compare pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if any(not _check_expr(context.annotated_expr[key]) for key in ["input_0", "input_1"]):
        return False
    if not _check_expr(context.annotated_expr["out"], ("bool")):
        return False
    ndim_a = len(context.annotated_expr["input_0"].struct_info.shape.values)
    ndim_b = len(context.annotated_expr["input_1"].struct_info.shape.values)
    return ndim_a == ndim_b


def _elemwise_check(context: PatternCheckContext) -> bool:
    """Check if the elemwise pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if not _basic_check(context):
        return False
    ndim_a = len(context.annotated_expr["input_0"].struct_info.shape.values)
    ndim_b = len(context.annotated_expr["input_1"].struct_info.shape.values)
    return ndim_a == ndim_b


def _reshape_check(context: PatternCheckContext) -> bool:
    """Check if the reshape pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    dtypes = ("float32", "float16", "int32")
    if any(not _check_expr(context.annotated_expr[key], dtypes) for key in ["input_0", "out"]):
        return False
    return True


def _take_check(context: PatternCheckContext) -> bool:
    """Check if the take pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    if any(not _check_expr(context.annotated_expr[key]) for key in ["input_0", "out"]):
        return False
    return _check_expr(context.annotated_expr["input_1"], ("int32"))


def _plugin_check(context: PatternCheckContext) -> bool:
    """Check if the plugin pattern is correct.

    Returns
    -------
    pass: bool
        Whether the pattern is correct.
    """

    ext_func = context.annotated_expr["out"].args[0]
    return bool(_ffi_api.IsPlugin(ext_func.global_symbol))


def plugin_attrs_getter(
    annotated_expr: Dict[str, tvm.relax.Expr],
) -> Dict[str, str]:
    """Get attributes for plugin pattern

    Parameters
    ----------
    annotated_expr: dict<str,Expr>
        The annotated exprs during fus pattern
    anchor: str
        The anchor key of expr

    Returns
    -------
    attrs: dict<str,str>
        The extra attributes for msc.
    """

    attrs = msc_pattern.msc_attrs_getter(annotated_expr, anchor="out")
    ext_func = annotated_expr["out"].args[0]
    attrs[_ffi_api.ToAttrKey("optype")] = ext_func.global_symbol
    return attrs


def wrap_basic_check(
    func: Callable[[PatternCheckContext], bool]
) -> Callable[[PatternCheckContext], bool]:
    """Wrapper a checker with basic check

    Returns
    -------
    checker: PatternCheckContext
        The wrapped checker.
    """

    @wraps(func)
    def wrapper(context):
        if not _basic_check(context):
            return False
        return func(context)

    return wrapper


CheckFunc = Callable[[Mapping[pattern.DFPattern, relax.Expr], relax.Expr], bool]
GetterFunc = Callable[[Mapping[pattern.DFPattern, relax.Expr], relax.Expr], Dict[str, str]]
Pattern = Union[
    FusionPattern,
    Tuple[str, pattern.DFPattern],
    Tuple[str, pattern.DFPattern, Mapping[str, pattern.DFPattern]],
    Tuple[str, pattern.DFPattern, Mapping[str, pattern.DFPattern], CheckFunc],
    Tuple[str, pattern.DFPattern, Mapping[str, pattern.DFPattern], CheckFunc, GetterFunc],
]


def get_patterns(target) -> List[Pattern]:
    """Get all the tensorrt patterns.

    Parameters
    ----------
    target: str
        The target name for tensorrt patterns.

    Returns
    -------
    patterns: list<Pattern>
        The patterns
    """

    basic_ops = {
        "nn.adaptive_avg_pool2d": ["input"],
        "nn.avg_pool2d": ["input"],
        "nn.conv2d": ["input", "constant"],
        "nn.max_pool2d": ["input"],
        "concat": ["input"],
        "clip": ["input", "input", "input"],
        "image.resize2d": ["input", "input"],
        "matmul": ["input", "input"],
        "permute_dims": ["input"],
        "strided_slice": ["input"],
    }
    activation_ops = ["nn.relu", "nn.softmax", "sigmoid", "tanh"]
    reduce_ops = ["max", "min", "mean", "sum"]
    unary_ops = ["cos", "exp", "negative", "round", "sin", "square", "sqrt", "tan"]
    elemwise_ops = [
        "add",
        "divide",
        "floor_divide",
        "maximum",
        "minimum",
        "multiply",
        "power",
        "subtract",
    ]
    compare_ops = ["greater", "less"]
    patterns = []
    # basic ops
    for op, in_types in basic_ops.items():
        inputs = ["input_" + str(i) for i in range(len(in_types))]
        patterns.append(
            (
                target + "." + op,
                *basic_pattern("relax." + op, in_types),
                _basic_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=inputs),
            )
        )
    # activation ops
    for op in activation_ops:
        patterns.append(
            (
                target + "." + op,
                *basic_pattern("relax." + op, ["input"]),
                _basic_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input_0"]),
            )
        )
    # reduce ops
    for op in reduce_ops:
        patterns.append(
            (
                target + "." + op,
                *basic_pattern("relax." + op, ["input"]),
                _basic_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input_0"]),
            )
        )
    # unary ops
    for op in unary_ops:
        patterns.append(
            (
                target + "." + op,
                *basic_pattern("relax." + op, ["input"]),
                _basic_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input_0"]),
            )
        )
    # elemwise ops
    for op in elemwise_ops:
        patterns.append(
            (
                target + "." + op,
                *elemwise_pattern("relax." + op),
                _elemwise_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input_0", "input_1"]),
            )
        )
    # compare ops
    for op in compare_ops:
        patterns.append(
            (
                target + "." + op,
                *elemwise_pattern("relax." + op),
                _compare_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input_0", "input_1"]),
            )
        )

    # special ops
    patterns.extend(
        [
            (
                target + ".take",
                *basic_pattern("relax.take", ["input", "input"]),
                _take_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input_0", "input_1"]),
            ),
            (
                target + ".argmax",
                *argmaxmin_pattern("relax.argmax"),
                _argmaxmin_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input"]),
            ),
            (
                target + ".argmin",
                *argmaxmin_pattern("relax.argmin"),
                _argmaxmin_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input"]),
            ),
            (
                target + ".reshape",
                *basic_pattern("relax.reshape", ["input", "input"]),
                _reshape_check,
                partial(msc_pattern.msc_attrs_getter, anchor="out", inputs=["input_0"]),
            ),
        ]
    )
    # fusable ops
    patterns.extend(
        [
            (
                target + ".msc.conv2d_bias",
                *msc_pattern.make_opt_relax_conv_bias_pattern("relax.nn.conv2d"),
                wrap_basic_check(msc_pattern._check_opt_relax_conv_bias),
                partial(
                    msc_pattern.msc_attrs_getter, anchor="conv", inputs=["data", "weight", "bias"]
                ),
            ),
        ]
    )
    # plugin ops
    patterns.append(
        (
            target + ".plugin",
            *basic_pattern("relax.call_dps_packed", ["input", "input"]),
            _plugin_check,
            plugin_attrs_getter,
        )
    )

    return patterns


register_patterns(get_patterns("msc_tensorrt"))
