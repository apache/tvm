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
# pylint: disable=unused-argument, not-context-manager
"""Utilities for partitioning input quantization and output dequantization expressions."""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor

# operators that are allowed in prefix/suffix partitions, because they are used
# to quantize/dequantize
ALLOWED_CONVERSION_OPS = ["add", "multiply", "right_shift", "clip", "round", "cast"]


def partition_conversions(mod, quantized_dtypes, ensure_fully_integral):
    """Partition mod into input quantization, core quantized inference, and output dequantization.

    The resulting module includes an additional `main` that fuses all three
    partitions together.

    Parameters
    ----------
    mod : tvm.IRModule
        Quantized module to partition

    quantized_dtypes : Set[str]
        Set of data types allowed in quantized operators

    ensure_fully_integral : bool
        Whether to raise an exception if there are unquantized operators in the result

    Returns
    -------
    fused_mod : tvm.IRModule
        Module containing the input quantization (`quantize_inputs`), core
        quantized inference (`quantized_main`), output dequantization
        (`dequantize_outputs`), and full quantized inference functions
    """
    # Partitioning is implemented as in the diagram below:
    #
    #   +----------------------------+
    #   |Quantized Inference Function|
    #   +--------------+-------------+
    #                  |
    #           partition_prefix
    #                  |
    #            +-----+-------------------------+
    #            |                               |
    #   +--------v---------+   +-----------------v------------------+
    #   |Input Quantization|   |Rest of Quantized Inference Function|
    #   +------------------+   +-----------------+------------------+
    #                                            |
    #                                    partition_suffix
    #                                            |
    #                                     +------+---------------------+
    #                                     |                            |
    #   +------------------+   +----------v------------+   +-----------v---------+
    #   |Input Quantization|   |Core Quantized Function|   |Output Dequantization|
    #   +------------------+   +-----------------------+   +---------------------+
    #
    # The final module contains all three partitions, as well as a
    # `main` function that composes these three functions (depicted below).
    #
    # +--------------------+-------------------------+-----------------------+
    # | Input Quantization | Core Quantized Function | Output Dequantization |
    # +--------------------+-------------------------+-----------------------+
    assert len(mod.functions) == 1
    pre_mod, mid_mod = partition_prefix(mod, quantized_dtypes)
    mid_mod, post_mod = partition_suffix(mid_mod, quantized_dtypes)
    if ensure_fully_integral:
        assert has_only_conversion_ops(pre_mod["main"])
        assert relay.analysis.all_dtypes(mid_mod["main"]).issubset(quantized_dtypes)
        assert has_only_conversion_ops(post_mod["main"])
    return fuse_partitions(pre_mod, mid_mod, post_mod)


def fuse_partitions(pre_mod, mid_mod, post_mod):
    """Combine prefix, middle, and suffix modules into a single module.

    The combined module includes an additional `main` that fuses all three
    partitions together.

    Parameters
    ----------
    pre_mod : tvm.IRModule
        Module containing an input quantization function

    mid_mod : tvm.IRModule
        Module containing core of a quantized inference function

    post_mod : tvm.IRModule
        Module containing an output dequantization function

    Returns
    -------
    fused_mod : tvm.IRModule
        Module containing the input quantization, core quantized inference,
        output dequantization, and full quantized inference functions
    """
    pre_func = pre_mod["main"]
    mid_func = mid_mod["main"]
    post_func = post_mod["main"]
    # create a module containing the prefix, middle, and suffix partitions
    fused_mod = tvm.IRModule(
        functions={
            relay.GlobalVar("quantize_inputs"): pre_func,
            relay.GlobalVar("quantized_main"): mid_func,
            relay.GlobalVar("dequantize_outputs"): post_func,
        }
    )

    # construct a `main` that strings together the partitions, such that its
    # behaviour is equivalent to `main` in an *unpartitioned* module
    scope_builder = relay.ScopeBuilder()
    fused_mod_main_params = [relay.Var(param.name_hint) for param in pre_func.params]
    quantized_inputs = scope_builder.let(
        "quantized_inputs",
        relay.Call(fused_mod.get_global_var("quantize_inputs"), fused_mod_main_params),
    )
    quantized_outputs = scope_builder.let(
        "quantized_outputs",
        relay.Call(
            fused_mod.get_global_var("quantized_main"),
            [relay.TupleGetItem(quantized_inputs, i) for i in range(len(pre_func.ret_type.fields))],
        ),
    )
    dequantized_outputs = scope_builder.let(
        "dequantized_outputs",
        relay.Call(fused_mod.get_global_var("dequantize_outputs"), [quantized_outputs]),
    )
    scope_builder.ret(dequantized_outputs)
    fused_mod["main"] = relay.Function(fused_mod_main_params, scope_builder.get())
    return relay.transform.InferType()(fused_mod)


class PrefixCutter(ExprMutator):
    """A mutator for extracting input quantization expressions from a function

    The result of `visit` is the core function, and the input quantization
    expressions are stored in the `prefix_sb` scope builder.
    """

    def __init__(self, params, quantized_dtypes):
        ExprMutator.__init__(self)
        self.params = set(params)
        self.quantized_dtypes = quantized_dtypes
        self.subtree_params = set()
        self.new_func_params = []
        self.prefix_sb = relay.ScopeBuilder()
        self.prefix_binding_map = {}

    def visit_var(self, var):
        if var in self.params:
            self.subtree_params.add(var)
        return var

    def visit_call(self, call):
        # TODO(weberlo) use graph pattern matching?
        if not hasattr(call.op, "name") or call.op.name not in ALLOWED_CONVERSION_OPS:
            new_args = []
            for arg in call.args:
                new_arg = self.visit(arg)
                if len(self.subtree_params) == 0:
                    new_args.append(new_arg)
                else:
                    assert len(self.subtree_params) == 1
                    param = next(iter(self.subtree_params))
                    pre_param = self.prefix_sb.let(param.name_hint, new_arg)
                    self.subtree_params.clear()
                    mid_param = relay.Var(param.name_hint, arg.checked_type)
                    self.prefix_binding_map[mid_param] = pre_param
                    # return new parameter, then we can use
                    # relay.analysis.free_vars at the end of the pass to generate
                    # new `mid_func` type signature
                    new_args.append(mid_param)
            return relay.Call(call.op, new_args, call.attrs)

        return super().visit_call(call)


def partition_prefix(mod, quantized_dtypes):
    """Extract input quantization expressions from `mod['main']`.

    Parameters
    ----------
    mod : tvm.IRModule
        Module containing a quantized inference function

    quantized_dtypes : Set[str]
        Set of data types allowed in quantized operators

    Returns
    -------
    pre_mod : tvm.IRModule
        Module containing the input quantization function

    mid_mod : tvm.IRModule
        Module containing a function with everything except for input quantization
    """
    assert len(mod.functions) == 1
    func = mod["main"]
    prefix_cutter = PrefixCutter(func.params, quantized_dtypes)
    mid_body = prefix_cutter.visit(func.body)
    assert not func.type_params, "unimplemented"
    assert not func.attrs, "unimplemented"
    mid_func = relay.Function(relay.analysis.free_vars(mid_body), mid_body)
    mid_mod = tvm.IRModule.from_expr(mid_func)
    mid_mod = relay.transform.InferType()(mid_mod)

    scope_builder = prefix_cutter.prefix_sb
    # make sure we pass through all inputs in the prefix function's return expr
    # (even those that don't require quantization)
    ret_expr = []
    for param in mid_func.params:
        if param in prefix_cutter.prefix_binding_map:
            # this param required a conversion, so we collected it in the
            # prefix cutter pass, and we can use the pass's mapping from mid
            # func params to pre func params
            ret_expr.append(prefix_cutter.prefix_binding_map[param])
        else:
            # there was no detected conversion for this argument, so we thread
            # it through the prefix function untouched
            ret_expr.append(relay.Var(param.name_hint, param.checked_type))
    ret_expr = relay.Tuple(ret_expr)
    scope_builder.ret(ret_expr)
    pre_func_body = scope_builder.get()
    pre_func = relay.Function(relay.analysis.free_vars(pre_func_body), pre_func_body)
    pre_mod = tvm.IRModule.from_expr(pre_func)
    pre_mod = relay.transform.InferType()(pre_mod)

    return pre_mod, mid_mod


class SuffixCutter(ExprMutator):
    """A mutator for extracting output dequantization expressions from a function

    The result of `visit` is a function containing the output dequantization
    expressions, and the middle of the function is stored in `mid_body`.
    """

    def __init__(self, quantized_dtypes):
        ExprMutator.__init__(self)
        self.mid_body = None
        self.quantized_dtypes = quantized_dtypes

    def visit(self, expr):
        if hasattr(expr, "checked_type") and expr.checked_type.dtype in self.quantized_dtypes:
            self.mid_body = expr
            return relay.Var("input", expr.checked_type)

        return super().visit(expr)


def partition_suffix(mod, quantized_dtypes):
    """Extract output dequantization expressions from `mod['main']`.

    Parameters
    ----------
    mod : tvm.IRModule
        Module containing a quantized inference function

    quantized_dtypes : Set[str]
        Set of data types allowed in quantized operators

    Returns
    -------
    pre_mod : tvm.IRModule
        Module containing the input quantization function

    mid_mod : tvm.IRModule
        Module containing a function with everything except for input quantization
    """
    assert len(mod.functions) == 1
    func = mod["main"]
    suffix_cutter = SuffixCutter(quantized_dtypes)
    post_body = suffix_cutter.visit(func.body)
    assert not func.type_params, "unimplemented"
    assert not func.attrs, "unimplemented"
    post_func = relay.Function(relay.analysis.free_vars(post_body), post_body, func.ret_type)
    post_mod = tvm.IRModule.from_expr(post_func)
    post_mod = relay.transform.InferType()(post_mod)

    mid_body = suffix_cutter.mid_body
    if mid_body is None:
        # The suffix contains the entire function, meaning there was no
        # quantization boundary in the given mod.  In this case, we use the
        # suffix mod as the middle mod and make the suffix an identity function.
        mid_mod = post_mod
        post_body = relay.Var("input", mid_mod["main"].ret_type)
        post_func = relay.Function([post_body], post_body)
        post_mod = tvm.IRModule.from_expr(post_func)
        post_mod = relay.transform.InferType()(post_mod)
    else:
        mid_func = relay.Function(func.params, mid_body)
        mid_mod = tvm.IRModule.from_expr(mid_func)
        mid_mod = relay.transform.InferType()(mid_mod)

    return mid_mod, post_mod


class ConversionOpChecker(ExprVisitor):
    """A pass for checking that the visited function contains only conversion ops"""

    def __init__(self):
        ExprVisitor.__init__(self)
        self.valid = True

    def visit_call(self, call):
        if not hasattr(call.op, "name") or call.op.name not in ALLOWED_CONVERSION_OPS:
            self.valid = False
        super().visit_call(call)


def has_only_conversion_ops(func):
    """Return true iff the given function contains only quantization/dequantization ops.

    Parameters
    ----------
    func : relay.Function
        Function being checked

    Returns
    -------
    valid : bool
        Whether the function contains only conversion ops
    """
    checker = ConversionOpChecker()
    checker.visit(func)
    return checker.valid
