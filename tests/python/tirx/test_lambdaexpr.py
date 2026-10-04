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
"""Lambda expressions retain lexical bindings inside typed tile operands."""

import pickle

import pytest
import tvm_ffi

import tvm
from tvm.ir import Expr, MissingType, Tuple, assert_structural_equal
from tvm.tirx import LambdaExpr, Var


def test_lambda_is_expr_and_roundtrips():
    lam = LambdaExpr(lambda x: x + 1)
    assert isinstance(lam, Expr)
    assert isinstance(lam.ty, MissingType)
    assert int(tvm.sym.Analyzer().simplify(lam.apply([4]))) == 5
    other = LambdaExpr(lambda y: y + 1)
    assert_structural_equal(lam, other)
    assert tvm_ffi.structural_hash(lam) == tvm_ffi.structural_hash(other)
    assert_structural_equal(lam, pickle.loads(pickle.dumps(lam)))
    assert tvm_ffi.structural_map(lam).same_as(lam)


@pytest.mark.parametrize("order", ["pre", "post"])
def test_lambda_remaps_binders_and_captures(order):
    capture = Var("capture", "int32")
    replacement = Var("replacement", "int32")
    lam = LambdaExpr(lambda x: x + capture)
    renamed = Var("renamed", "int32")

    def rewrite(var, kind):
        if var.same_as(lam.vars[0]):
            return renamed
        if var.same_as(capture):
            return replacement
        return var

    result = tvm_ffi.structural_map(lam, with_def_region_kind=(Var, rewrite), order=order)
    assert result.vars[0].same_as(renamed)
    assert_structural_equal(result.pred, renamed + replacement)
    assert_structural_equal(lam.pred, lam.vars[0] + capture)


def test_lambda_remap_does_not_escape_its_body():
    lam = LambdaExpr(lambda x: x + 1)
    x = lam.vars[0]
    renamed = Var("renamed", "int32")
    result = tvm_ffi.structural_map(
        Tuple([lam, x]),
        with_def_region_kind=(Var, lambda var, kind: renamed if kind else var),
    )
    assert result[0].vars[0].same_as(renamed)
    assert result[1].same_as(x)


def test_lambda_scope_and_tile_script_roundtrip():
    from tvm.script import tirx as T

    @T.prim_func
    def func(A: T.Tensor((8,), "float32"), n: T.int32):
        T.tile.select(A[:], A[:], T.float32(0), lambda x: x < n)

    assert tvm.tirx.analysis.verify_well_formed(func)
    assert_structural_equal(func, tvm.script.from_source(func.script(), extra_vars={"T": T}))
    lam = LambdaExpr(lambda x: x + 1)
    leaked = tvm.tirx.PrimFunc(
        [], tvm.tirx.SeqStmt([tvm.tirx.Evaluate(lam), tvm.tirx.Evaluate(lam.vars[0])])
    )
    with pytest.raises((ValueError, tvm.error.InternalError), match="undefined|in-scope"):
        tvm.tirx.analysis.verify_well_formed(leaked)


def test_lambda_shadows_existing_remap_and_substitutes_free_capture():
    capture = Var("capture", "int32")
    lam = LambdaExpr(lambda x: x + capture)
    x = lam.vars[0]
    outer_x = Var("outer_x", "int32")
    new_capture = Var("new_capture", "int32")

    def substitute(root, mutator):
        mutator.var_remap_set(x, outer_x)
        mutator.var_remap_set(capture, new_capture)
        return mutator.default_mutate(root)

    result = tvm_ffi.structural_mutate(Tuple([x, lam, x]), (Tuple, substitute))
    assert result[0].same_as(outer_x) and result[2].same_as(outer_x)
    assert result[1].vars[0].same_as(x)
    assert_structural_equal(result[1].pred, x + new_capture)
