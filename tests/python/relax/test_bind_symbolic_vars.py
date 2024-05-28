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

import pytest

import tvm
import tvm.testing
from tvm.script import relax as R, tir as T

replace_by_tir_var = tvm.testing.parameter(
    by_dict={"replace-by-string": False, "replace-by-tir-var": True}
)


def test_bind_static_value(replace_by_tir_var):
    """Symbolic vars may be replaced

    The replaced variables may be given either as strings, or as TIR variables
    """

    @R.function(private=True)
    def before(A: R.Tensor(("M", "K")), B: R.Tensor(("K", "N"))) -> R.Tensor(("M", "N")):
        return R.matmul(A, B)

    @R.function(private=True)
    def expected(A: R.Tensor((128, 64)), B: R.Tensor((64, 32))) -> R.Tensor((128, 32)):
        return R.matmul(A, B)

    if replace_by_tir_var:
        M, K = before.params[0].struct_info.shape
        _, N = before.params[1].struct_info.shape
        symbolic_var_map = {M: 128, K: 64, N: 32}
    else:
        symbolic_var_map = {"M": 128, "K": 64, "N": 32}

    after = before.bind_symbolic_vars(symbolic_var_map)
    tvm.ir.assert_structural_equal(expected, after)


def test_error_with_duplicate_var_names():
    """Duplicate variable names may not be replaced by string

    Two TIR variables may have the same name.  If two symbolic
    variables share the same name, the replacement map may not refer
    to that variable by string.
    """
    N1 = tvm.tir.Var("N", "int64")
    N2 = tvm.tir.Var("N", "int64")

    @R.function(private=True)
    def func(A: R.Tensor((N1, N1)), B: R.Tensor((N1, N2))) -> R.Tensor((N1, N2)):
        out: R.Tensor((N1, N2)) = R.matmul(A, B)
        return out

    with pytest.raises(tvm.TVMError):
        func.bind_symbolic_vars({"N": 64})


def test_string_var_when_other_var_has_duplicate_var_names():
    """Like test_error_with_duplicate_var_names, but replacing a different variable

    If two TIR variables share the same name, the restriction against
    replacing variables by name only applies to those duplicate names.
    Other variables may still be replaced by name.
    """
    N1 = tvm.tir.Var("N", "int64")
    N2 = tvm.tir.Var("N", "int64")
    BatchSize = tvm.tir.Var("BatchSize", "int64")

    @R.function(private=True)
    def before(
        A: R.Tensor((BatchSize, N1, N1)), B: R.Tensor((N1, N2))
    ) -> R.Tensor((BatchSize, N1, N2)):
        out: R.Tensor((BatchSize, N1, N2)) = R.matmul(A, B)
        return out

    @R.function(private=True)
    def expected(A: R.Tensor((16, N1, N1)), B: R.Tensor((N1, N2))) -> R.Tensor((16, N1, N2)):
        out: R.Tensor((16, N1, N2)) = R.matmul(A, B)
        return out

    after = before.bind_symbolic_vars({"BatchSize": 16})
    tvm.ir.assert_structural_equal(expected, after)


def test_error_with_nonexisting_var_name():
    """A string name of a symbolic var must be used by the function"""

    @R.function(private=True)
    def func(A: R.Tensor(("M", "N"))):
        return A

    with pytest.raises(tvm.TVMError):
        func.bind_symbolic_vars({"non_existing_symbolic_var": 64})


def test_error_with_nonexisting_tir_var():
    """A TIR symbolic var must be a symbolic var of the function"""

    @R.function(private=True)
    def func(A: R.Tensor(["M", "N"])):
        return A

    with pytest.raises(tvm.TVMError):
        func.bind_symbolic_vars({tvm.tir.Var("M", "int64"): 64})


def test_error_with_multiple_definitions():
    """The string/TIR var syntaxes may not define the same variable"""

    @R.function(private=True)
    def func(A: R.Tensor(["M", "N"])):
        return A

    tir_var = func.params[0].struct_info.shape[0]
    symbolic_var_map = {tir_var: 0, "M": 0}

    with pytest.raises(tvm.TVMError):
        func.bind_symbolic_vars(symbolic_var_map)


def test_error_if_output_has_undefined():
    """The replacements may not introduce undefined symbolic vars"""

    @R.function(private=True)
    def func(A: R.Tensor(["M", "N"])):
        return A

    outside_var = tvm.tir.Var("outside_var", "int64")

    with pytest.raises(tvm.TVMError):
        func.bind_symbolic_vars({"M": outside_var * 2})


def test_replacements_may_produce_new_symbolic_vars():
    """The output may introduce symbolic vars, but they must be bound"""

    @R.function(private=True)
    def before(A: R.Tensor(["M", "N"])):
        return A

    @R.function(private=True)
    def expected(A: R.Tensor(["outside_var * 2", "outside_var"])):
        return A

    outside_var = tvm.tir.Var("outside_var", "int64")

    after = before.bind_symbolic_vars({"M": outside_var * 2, "N": outside_var})
    tvm.ir.assert_structural_equal(expected, after)


def test_bind_symbolic_vars_in_tensor_shape():
    """The bound variable should be replaced when appearing in struct info"""

    @R.function(private=True)
    def before(A: R.Tensor(["M", "N"])):
        M = T.int64()
        N = T.int64()
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([2 * M * N]))
        return B

    @R.function(private=True)
    def expected(A: R.Tensor(["M", 16])):
        M = T.int64()
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([M * 32]))
        return B

    after = before.bind_symbolic_vars({"N": 16})
    tvm.ir.assert_structural_equal(expected, after)


def test_bind_symbolic_vars_in_shape_expr():
    """The bound variable should be replaced when appearing in R.Shape"""

    @R.function(private=True)
    def before(A: R.Tensor(["M * N"]), x: R.Shape(["M", "N"])):
        M = T.int64()
        N = T.int64()
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([2 * M * N]))
        return B

    @R.function(private=True)
    def expected(A: R.Tensor(["M * 16"]), x: R.Shape(["M", 16])):
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([M * 32]))
        return B

    after = before.bind_symbolic_vars({"N": 16})
    tvm.ir.assert_structural_equal(expected, after)


def test_bind_defining_of_symbolic_vars_in_prim_value():
    """R.Prim may define symbolic variables

    This case is a bit odd, because it always results in a
    fully-constrained parameter at the relax level.  After binding in
    this test case, we have a function that accepts three parameters,
    and the third parameter must always be the number 16.

    However, this provides the most consistent behavior with other
    uses of `relax.Function.bind_symbolic_vars`, which restricts the
    allowed values for each parameter, but does not alter the number
    of parameters.  This is in contrast to the `BindParams` pass,
    which provides a known value for relax parameters, removing them
    from the function signature.

    This convention also prevents surprise changes to the function
    signature, such as shown in
    `test_bind_symbolic_vars_with_expr_in_prim_value`.
    """

    @R.function(private=True)
    def before(A: R.Tensor(["M * N"]), x: R.Prim(value="M"), y: R.Prim(value="N")):
        M = T.int64()
        N = T.int64()
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([2 * M * N]))
        return B

    @R.function(private=True)
    def expected(A: R.Tensor(["M * 16"]), x: R.Prim(value="M"), y: R.Prim(value=16)):
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([M * 32]))
        return B

    after = before.bind_symbolic_vars({"N": 16})
    tvm.ir.assert_structural_equal(expected, after)


def test_bind_usage_of_symbolic_vars_in_prim_value():
    """R.Prim may use symbolic variables defined by other parameters

    Like test_bind_defining_of_symbolic_vars_in_prim_value, but with
    R.Prim using a symbolic variable rather than defining it.

    This also demonstrates why we should not remove fully-constrained
    R.Prim function parameters.  In this case, we have a function that
    accepts two parameters, and we have specialized the shape of the
    first parameter.  It would be unexpected for specialization of the
    first parameter to result in removal of a different parameter
    altogether.
    """

    @R.function(private=True)
    def before(A: R.Tensor(["M", "N"]), x: R.Prim(value="M*N")):
        M = T.int64()
        N = T.int64()
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([2 * M * N]))
        return B

    @R.function(private=True)
    def expected(A: R.Tensor([16, 16]), x: R.Prim(value=256)):
        B = R.call_dps_packed("dummy_func", [A], out_sinfo=R.Tensor([512]))
        return B

    after = before.bind_symbolic_vars({"M": 16, "N": 16})
    tvm.ir.assert_structural_equal(expected, after)


def test_bind_strided_slice():
    """relax.op.strided_slice stores PrimExpr attributes"""

    @R.function(private=True)
    def before(A: R.Tensor(["M", "N"])):
        N = T.int64()
        B = R.strided_slice(A, [1], [0], [N // 4])
        return B

    @R.function(private=True)
    def expected(A: R.Tensor(["M", 32])):
        B = R.strided_slice(A, [1], [0], [8])
        return B

    after = before.bind_symbolic_vars({"N": 32})
    tvm.ir.assert_structural_equal(expected, after)


def test_bind_inside_match_cast():
    """Symbolic variables may occur within R.match_cast"""

    @R.function(private=True)
    def before(A: R.Tensor(["M", "N"]), B: R.Tensor(ndim=2)):
        M = T.int64()
        N = T.int64()
        C = R.match_cast(B, R.Tensor([M, N]))
        D = R.add(A, C)
        return D

    @R.function(private=True)
    def expected(A: R.Tensor(["M", 32]), B: R.Tensor(ndim=2)):
        M = T.int64()
        C = R.match_cast(B, R.Tensor([M, 32]))
        D = R.add(A, C)
        return D

    after = before.bind_symbolic_vars({"N": 32})
    tvm.ir.assert_structural_equal(expected, after)


if __name__ == "__main__":
    tvm.testing.main()
