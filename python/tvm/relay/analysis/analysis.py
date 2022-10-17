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
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
"""
This file contains the set of passes for Relay, which exposes an interface for
configuring the passes and scripting them in Python.
"""
from ...ir import IRModule
from ...relay import transform, build_module
from ...runtime.ndarray import cpu

from . import _ffi_api
from .feature import Feature


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fvisit : function
        The visitor function to be applied.
    """
    return _ffi_api.post_order_visit(expr, fvisit)


def well_formed(expr):
    """Check that each Var is only bound once (well formed).

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    well_form : bool
        Whether the input expression is well formed
    """
    return _ffi_api.well_formed(expr)


def check_kind(t, mod=None):
    """Check that the type is well kinded and return the kind.
    For example, this mean type cannot has tensor of tensor, or is a tuple type
    of 2 shapes.

    Parameters
    ----------
    t : tvm.relay.Type
        The type to check

    mod : Optional[tvm.IRModule]
        The global module.

    Returns
    -------
    kind : Kind
        the kind of t

    Examples
    --------
    .. code:: python

        assert check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Shape)])) == Shape
        assert check_kind(relay.TupleType([relay.TypeParam('tp1', relay.Kind.Type)])) == Type
    """
    if mod is not None:
        return _ffi_api.check_kind(t, mod)
    else:
        return _ffi_api.check_kind(t)


def check_constant(expr):
    """Check whether an expression is constant

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    result : bool
        Whether the expression is constant.
    """
    return _ffi_api.check_constant(expr)


def check_basic_block_normal_form(expr):
    """Check whether an expression is in the basic block form

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    result : bool
        Whether the expression is in the basic block form.
    """
    return _ffi_api.check_basic_block_normal_form(expr)


def free_vars(expr):
    """Get free Vars from expression expr in Post DFS order.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of free variables in post DFS order.

    Note
    ----
    The fact that Vars are post-DFS ordred are useful in
    neural networks: usually this means weights of previous
    are ordered first.
    """
    return _ffi_api.free_vars(expr)


def bound_vars(expr):
    """Get bound vars from expression expr in post-DFS order.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of bound variables in post-DFS order.
    """
    return _ffi_api.bound_vars(expr)


def all_vars(expr):
    """Get all vars from expression expr in post-DFS order.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    free : List[tvm.relay.Var]
        The list of all variables in post-DFS order.
    """
    return _ffi_api.all_vars(expr)


def free_type_vars(expr, mod=None):
    """Get free type variables from expression/type e

    Parameters
    ----------
    expr : Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type

    mod : Optional[tvm.IRModule]
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of free type variables in post-DFS order
    """
    use_mod = mod if mod is not None else IRModule()
    return _ffi_api.free_type_vars(expr, use_mod)


def bound_type_vars(expr, mod=None):
    """Get bound type variables from expression/type e

    Parameters
    ----------
    expr : Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type

    mod : Optional[tvm.IRModule]
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of bound type variables in post-DFS order
    """
    use_mod = mod if mod is not None else IRModule()
    return _ffi_api.bound_type_vars(expr, use_mod)


def all_type_vars(expr, mod=None):
    """Get all type variables from expression/type e

    Parameters
    ----------
    expr : Union[tvm.relay.Expr,tvm.relay.Type]
        The input expression/type

    mod : Optional[tvm.IRModule]
        The global module

    Returns
    -------
    free : List[tvm.relay.TypeVar]
        The list of all type variables in post-DFS order
    """
    use_mod = mod if mod is not None else IRModule()
    return _ffi_api.all_type_vars(expr, use_mod)


def all_dtypes(expr):
    """Collect set of all data types used in `expr`.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression

    Returns
    -------
    ret : Set[String]
        Set of data types used in the expression (e.g., `{'int8', 'int32'}`)
    """
    return set(_ffi_api.all_dtypes(expr))


def get_total_mac_number(expr):
    """
    Count the number of MACs (multiply-accumulate) of a model

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    result : int64
      The number of MACs (multiply-accumulate) of a model
    """
    return _ffi_api.GetTotalMacNumber(expr)


def unmatched_cases(match, mod=None):
    """
    Finds cases that the match expression does not catch, if any.

    Parameters
    ----------
    match : tvm.relay.Match
        The match expression

    mod : Optional[tvm.IRModule]
        The module (defaults to an empty module)

    Returns
    -------
    missing_patterns : [tvm.relay.Pattern]
        Patterns that the match expression does not catch.
    """
    return _ffi_api.unmatched_cases(match, mod)


def detect_feature(a, b=None):
    """
    Detect the feature used in a relay program.

    Parameters
    ----------
    a : Union[tvm.relay.Expr, tvm.IRModule]
      The input expression or module.

    b : Optional[Union[tvm.relay.Expr, tvm.IRModule]]
      The input expression or module.
      The two arguments cannot both be expression or module.

    Returns
    -------
    features : Set[Feature]
      Features used in the program.
    """
    if isinstance(a, IRModule):
        a, b = b, a
    return {Feature(int(x)) for x in _ffi_api.detect_feature(a, b)}


def extract_fused_functions(mod):
    """Pass to extract IRModule of only fused primitive functions.

    The ExtractFusedFunctions pass invokes SimplifyInference, FuseOps(3),
    and ExtractFusedFunctions in that order

    Parameters
    ----------
    mod : tvm.IRModule

    Returns
    -------
    ret : Dict[int, tvm.relay.function.Function]
        A module containing only fused primitive functions
    """
    ret_mod = _ffi_api.ExtractFusedFunctions()(mod)
    ret = {}
    for hash_, func in ret_mod.functions.items():
        ret[hash_] = func
    return ret


def list_op_freqs(mod):
    """Pass to extract unique operator names and how frequently they appear
    in an IRModule. Fused functions are traversed to count the operators
    that compose them.

    Parameters
    ----------
    mod : tvm.IRModule

    Returns
    -------
    ret : Dict[str, int]
        Dict of unique operator names to frequency
    """
    return _ffi_api.ExtractOperators(mod)


def list_fake_quantized_op_freqs(mod):
    """Pass to extract fake quantized op names and the frequency that they appear
    in fake quantized regions of an IRModule.

    Parameters
    ----------
    mod : tvm.IRModule

    Returns
    -------
    ret : Dict[str, int]
        Dict of fake quantized operator names to frequency
    """
    return _ffi_api.ExtractFakeQuantizedOps(mod)


def search_fc_transpose(expr):
    """Search fc weight name in the patten: y = nn.dense(x, transpose(w, [1, 0]))

    This function is used in the data_dep_optimization.simplify_fc_transpose method

    Parameters
    ----------
    expr : tvm.relay.Expr

    Returns
    -------
    ret : Array[String]
        Array of weight variable name in pattern y = nn.dense(x, transpose(w, [1, 0]))
    """
    ret = _ffi_api.search_fc_transpose(expr)
    return ret


def get_calibration_data(mod, data):
    """Get the calibration data of a given relay graph

    This pass uses the graph executor to get the calibration data of a module, which
    includes the input and output values of each function. The returned data uses
    the GlobalVar of each function as a key. Users can further access the inputs and
    outputs by using `inputs` or  `outputs` as the key.

    Following are some limitations:
    1. The input module (graph) cannot have control flows.
    2. The input arguments of each function cannot be tuples (outputs can be tuples).
    3. We only handle top-level functions (i.e., nested function is not handled).
    4. We only handle functions with `Compiler` attribute being set.

    Parameters
    ----------
    mod : tvm.IRModule
        The input module for collecting the calibration data

    data : Dict[str, NDArray]
        The input data for running the module

    Returns
    -------
    data : Dict[tvm.relay.GlobalVar, Dict[str, NDArray]]
    """
    output_map = _ffi_api.get_calibrate_output_map(mod)

    mod = _ffi_api.get_calibrate_module(mod)
    mod = transform.Inline()(mod)

    ref_res = build_module.create_executor("graph", mod=mod, device=cpu(0)).evaluate()(**data)

    calib_data = {}
    for gvar, indices in output_map.items():
        offset = int(indices[0])
        in_len = int(indices[1])
        out_len = int(indices[2])
        value = {
            "inputs": ref_res[offset : offset + in_len],
            "outputs": ref_res[offset + in_len : offset + in_len + out_len],
        }
        calib_data[gvar] = value

    return calib_data


def extract_intermdeiate_expr(mod, expr_id):
    """Extract Relay Expr by its expression ID

    This function is used for extracting Relay Expr
    by its expression ID of the main function
    that we can see in `print(mod["main"])`.

    Parameters
    ----------
    mod : tvm.IRModule

    expr_id : the Expr ID that we want to extract

    Returns
    -------
    ret : Extracted IRModule

    Examples
    --------
    .. code-block:: python

        # Suppose our module is printed like this:
        # def @main(%x: Tensor[(1, 1, 5, 1), float32], %w1, %w2) {
        #   %0 = nn.conv2d(%x, %w1, padding=[1, 1, 1, 1], channels=1, kernel_size=[3, 3]);
        #   %1 = nn.conv2d(%0, %w2, padding=[1, 1, 1, 1], channels=1, kernel_size=[3, 3]);
        #   %2 = add(%0, %1);
        #   %3 = split(%2, indices_or_sections=1);
        #   %4 = %3.0;
        #   add(%4, 1f)
        # }
        # if we want to extract `%1 = nn.conv2d`
        from tvm import relay

        relay.analysis.extract_intermdeiate_expr(mod, 1)
    """
    return _ffi_api.ExtractIntermediateExpr(mod, expr_id)
