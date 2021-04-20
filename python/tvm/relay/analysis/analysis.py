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
import tvm
from ...ir import IRModule
from ...relay import transform, build_module
from ...runtime.ndarray import cpu

from . import _ffi_api
from .feature import Feature


def context_analysis(mod, default_device):
    """Analyze the device context information of each IR node in a Relay
    program.

    Parameters
    ----------
    mod : tvm.IRModule
        The input module.

    default_device : tvm.runtime.Device
        The default context allocated to an IR node.
    """
    return _ffi_api.ContextAnalysis(mod, default_device)


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


def collect_device_info(expr):
    """Collect the device allocation map for the given expression. The device
    ids are propagated from the `device_copy` operators.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    ret : Dict[tvm.relay.ir.expr, int]
        A dictionary mapping tvm.relay.Expr to device type.
    """
    return _ffi_api.CollectDeviceInfo(expr)


def collect_device_annotation_ops(expr):
    """Collect the device annotation ops for the given expression.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    Returns
    -------
    ret : Dict[tvm.relay.Expr, int]
        A dictionary mapping tvm.relay.Expr to device type where the keys are
        annotation expressions.
    """
    return _ffi_api.CollectDeviceAnnotationOps(expr)


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


def pipeline_graph(expr, indices):
    """Split Graph Into A Group Of Subgraph
    Parameters
    ----------
    expr : tvm.relay.Expr
    indices : Array[int]

    Returns
    -------
    ret : Array[tvm.relay.IRModule]
    """

    def run_opt_pass(expr, opt_pass):
        """Exectue a relay pass"""
        assert isinstance(opt_pass, tvm.transform.Pass)
        mod = tvm.IRModule.from_expr(expr)
        mod = tvm.relay.transform.InferType()(mod)
        mod = opt_pass(mod)
        entry = mod["main"]
        return entry if isinstance(expr, tvm.relay.Function) else entry.body

    def _operator_idx_inc(expr, operator_current_idx):
        """Increase operator index"""
        if not isinstance(expr, tvm.relay.expr.Constant):
            operator_current_idx = operator_current_idx + 1

        return operator_current_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        # Parameters
        # ----------
        # constant_expr:
        #     constant expression
        # expr:
        #     expression to merge with constant expression

        # If body not let, then reached end of the express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, operator_indx, pipeline_mods, indices, constant_expr):
        # Enumrate all operator of compute graph then split the compute graph
        # into a group subgraph.
        # Parameters
        # ----------
        # anf:
        #     ANF format expression
        # operator_indx:
        #     current operator indice
        # pipeline_mods:
        #     the subgraph list get storage in this variable
        # indices:
        #     Array of indices use to define the subgraph scope
        # constant_expr:
        #     constant defined before current operator

        # Do the split work
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, operator_indx, pipeline_mods, indices, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            operator_indx = _operator_idx_inc(value, operator_indx)

            # record constan expr to make sure all sugraph can find correct
            # constant.
            if isinstance(value, tvm.relay.expr.Constant):
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)

            if isinstance(value, tvm.relay.expr.Call):
                if isinstance(value.op, tvm.ir.Op):

                    # if have expr a(b(c(d(e)))) and indexes are [1,2,3]
                    # then would get separate modules for a(b),c,d(e).
                    # the split area is a(b)[0,1] c[2,2] d(e)[2,3]
                    if indices and operator_indx == indices[0]:
                        indices.pop(0)
                        ann = _recursion(
                            anf.body, operator_indx, pipeline_mods, indices, constant_expr
                        )

                        # when current subgraph use previous subgraph constant,
                        # such constant may become free varaible due to the constant
                        # not exist, merge the previous constant with current subgraph
                        # to avoid such issue.
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)

                        ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, mod)
                        return tvm.relay.expr.Let(anf.var, value, anf.var)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, operator_indx, pipeline_mods, indices, constant_expr),
            )
        else:
            return anf

    pipeline_mods = []

    # operator count start from 0, then initial value get set into -1
    operator_indx = -1
    constant_expr = None
    subgraph_indices = indices.copy()
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = _recursion(anf, operator_indx, pipeline_mods, subgraph_indices, constant_expr)
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    return pipeline_mods
