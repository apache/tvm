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
"""APIs for pattern-based rewriting."""
from typing import Dict, Callable
from .pattern import DFPattern
from .context import PatternContext

from ..expr import Expr, Function, Var
from . import _ffi as ffi


def rewrite_call(
    pattern: DFPattern, rewriter: Callable[[Expr, Dict[DFPattern, Expr]], Expr], func: Function
) -> Function:
    """
    Rewrite a function with the given pattern and the rewriter function.

    Parameters
    ----------
    pattern: DFPattern
        The pattern to match.

    rewriter: Callable[[Expr, Dict[DFPattern, Expr]], Expr]
        The function to be called on a successful matching for rewriting. Given the matched
        call node and the map of patterns and matched expressions, it should return a new call node
        to replace the original one or the original matched call node as is.

        For example, to replace x + x with 2 * x, we can write the rewriter as follows:
        ```
        x = wildcard()
        pattern = is_op("relax.add")(x, x)

        def rewriter(orig, matchings):
            return R.multiply(matchings[x], R.const(2, "float32"))
        ```

    func: Function
        The function to rewrite.

    Returns
    -------
    rewritten_func: Function
        The rewritten or the input function, depending on the pattern matching result.
    """
    return ffi.rewrite_call(pattern, rewriter, func)


def rewrite_bindings(
    ctx: PatternContext,
    rewriter: Callable[[Dict[DFPattern, Var], Dict[Var, Expr]], Dict[Var, Expr]],
    func: Function,
) -> Function:
    """
    Rewrite a function with the given pattern and the rewriter function.

    Parameters
    ----------
    ctx: PatternContext
        The pattern constraint context under which rewriting takes place.

    rewriter: Callable[[Dict[DFPattern, Var], Dict[Var, Expr]], Dict[Var, Expr]]
        The function to be called on a successful matching for rewriting. Given the map of patterns
        and corresponding variables (bound variables or parameters), it should return a map that
        specifies new values for matched bound variables. It can refer to the passed bindings to
        create the replacement expressions.

        For example, to rewrite three matmuls for QKV projection in transformer models into one
        matmul followed by slicing, one can use the follwoing rewriter:
        ```
        inp_pat = wildcard()
        Q_weight_pat, K_weight_pat, V_weight_pat = wildcard(), wildcard(), wildcard()

        matmul1 = is_op("relax.matmul")(inp_pat, Q_weight_pat)
        matmul2 = is_op("relax.matmul")(inp_pat, K_weight_pat)
        matmul3 = is_op("relax.matmul")(inp_pat, V_weight_pat)

        def rewriter(matchings):
            inp = matchings[inp_pat]
            Q_weight = matchings[Q_weight_pat]
            K_weight = matchings[K_weight_pat]
            V_weight = matchings[V_weight_pat]
            width = Q_weight.struct_info.shape[1]

            concat = R.concat([Q_weight, K_weight, V_weight], axis=1)
            matmul = R.matmul(inp, concat)
            Q = R.strided_slice(matmul, axes=[2], begin=[0], end=[width])
            K = R.strided_slice(matmul, axes=[2], begin=[width], end=[width * 2])
            V = R.strided_slice(matmul, axes=[2], begin=[width * 2], end=[width * 3])

            # matchings[matmul1] gives the bound variable in the binding whose RHS matches with
            # the matmul1 pattern. For example, lv0 in lv0 = R.matmul(x1, w0).
            # We want to replace the RHS of this binding with Q.
            return {matchings[matmul1]: Q, matchings[matmul2]: K, matchings[matmul3]: V}
        ```

    func: Function
        The function to rewrite.

    Returns
    -------
    rewritten_func: Function
        The rewritten or the input function, depending on the pattern matching result.
    """
    return ffi.rewrite_bindings(ctx, rewriter, func)
