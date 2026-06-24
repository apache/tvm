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
# ruff: noqa: E501, F841
"""
ONNX testcases
================
This file is a test script to test Relax ONNX frontend coverage.
"""

from typing import Literal

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx
import onnxruntime
import tvm_ffi
from onnx import ModelProto, TensorProto, helper, numpy_helper

import tvm
import tvm.testing
from tvm import relax, topi
from tvm.relax.frontend.onnx import from_onnx
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

bg = np.random.MT19937(0)
rg = np.random.Generator(bg)


def _shape_with_size_vars(shape, prefix):
    size_vars = {}
    result = []
    for i, dim in enumerate(shape):
        if isinstance(dim, str):
            if dim == "?":
                result.append(tvm.tirx.SizeVar(f"{prefix}_{i}", "int64"))
            else:
                result.append(size_vars.setdefault(dim, tvm.tirx.SizeVar(dim, "int64")))
        else:
            result.append(dim)
    return result


def generate_random_inputs(
    model: ModelProto, inputs: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    input_values = {}
    # Iterate through model inputs and extract their shape.
    for i in model.graph.input:
        if inputs is not None and i.name in inputs and inputs[i.name] is not None:
            input_values[i.name] = inputs[i.name]
            continue
        shape = []
        for dim in i.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)

        input_values[i.name] = generate_random_value(shape, i.type.tensor_type.elem_type)

    return input_values


def generate_random_value(shape, elem_type) -> np.ndarray:
    # Extract datatype for the input.
    if elem_type:
        dtype = str(helper.tensor_dtype_to_np_dtype(elem_type))
    else:
        dtype = "float32"

    # Generate random inputs for each input.
    if dtype == "bool":
        # random_value = np.random.choice(a=[False, True], size=shape)
        random_value = rg.choice(a=[False, True], size=shape)
    elif dtype.startswith("int"):
        # Keep non-zero values
        random_value = rg.integers(low=-63, high=63, size=shape).astype(dtype)
        random_value[random_value <= 0] -= 1
    else:
        random_value = rg.standard_normal(size=shape).astype(dtype)

    return random_value


def check_correctness(
    model: ModelProto,
    inputs: dict[str, np.ndarray] | None = None,
    ir_version: int = 8,
    opset: int = 14,
    rtol: float = 1e-7,
    atol: float = 1e-5,
    check_dtypes: bool = False,
) -> None:
    """Run an onnx model in both onnxruntime and TVM through our importer
       confirm that the results match. Otherwise, an exception will be raised.

    Parameters
    ----------
    model: ModelProto
        The input onnx model that should be tested.
    inputs: Optional[Dict[str, np.ndarray]]
        An optional dictionary containing values for each input in the onnx model.
    ir_version: int
        Which version of the onnx IR to use.
    opset: int
        The opset version to use for the onnx importer.
    atol: float
        Set the tolerance of correctness checking. Some ops may be show more
        arithmetic variance than others.
    check_dtypes: bool
        Check if data types are the same.
    """
    # Configure model format.
    if ir_version is not None:
        model.ir_version = ir_version
    if opset is not None:
        model.opset_import[0].version = opset

    # If inputs are not provided, extract them from the onnx graph and produce random
    # values that we'll use for testing.
    inputs = generate_random_inputs(model, inputs)

    # Run the model through onnx to get the expected result.
    ort_session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_output = ort_session.run([], inputs)

    # Convert the onnx model into relax through the onnx importer.
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    # Convert operators for inference mode.
    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    # Legalize any relax ops into tensorir.
    tvm_model = relax.transform.LegalizeOps()(tvm_model)

    # Separate model from parameters.
    tvm_model, params = relax.frontend.detach_params(tvm_model)
    # Compile the relax graph into a VM then run.
    with tvm.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())
    # Prepare inputs.
    input_list = [
        inputs[key.name_hint] for key in tvm_model["main"].params if key.name_hint in inputs
    ]
    if params:
        input_list += params["main"]

    # Run model and check outputs.
    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    # Wrap as a list if there is only one output.
    if len(ort_output) == 1:
        # Do not check the output number for TVM
        # As for sequence output, the TVM output is a Tuple
        # while the ONNX output number is one, which is a list
        tvm_output = [tvm_output]

    def _get_numpy_subdtype(narray):
        if np.issubdtype(narray.dtype, np.integer):
            return "integer"
        elif np.issubdtype(narray.dtype, np.floating):
            return "floating"
        elif np.issubdtype(narray.dtype, np.bool_):
            return "bool"
        elif np.issubdtype(narray.dtype, np.complexfloating):
            return "complexfloating"
        else:
            return "other"

    def _check_output(tvm_out, ort_out):
        if isinstance(tvm_out, tuple) and isinstance(ort_out, tvm_ffi.Shape | list):
            assert len(tvm_out) == len(ort_out), "Unequal number of outputs"
            for tvm_out_i, ort_out_i in zip(tvm_out, ort_out):
                _check_output(tvm_out_i, ort_out_i)
        elif isinstance(tvm_out, tvm.runtime.Tensor) and isinstance(ort_out, np.ndarray):
            if check_dtypes:
                assert tvm_out.numpy().dtype == ort_out.dtype
            tvm.testing.assert_allclose(tvm_out.numpy(), ort_out, rtol=rtol, atol=atol)
        elif isinstance(tvm_out, tvm_ffi.Shape) and isinstance(ort_out, np.ndarray):
            shape_out = tvm.runtime.tensor([int(i) for i in tvm_out])
            if check_dtypes:
                assert _get_numpy_subdtype(shape_out.numpy()) == _get_numpy_subdtype(ort_out)
            tvm.testing.assert_allclose(shape_out.numpy(), ort_out, rtol=rtol, atol=atol)
        elif isinstance(tvm_out, int | float | bool) and isinstance(ort_out, np.ndarray):
            if check_dtypes:
                assert _get_numpy_subdtype(np.array(tvm_out)) == _get_numpy_subdtype(ort_out)
            tvm.testing.assert_allclose(np.array(tvm_out), ort_out, rtol=rtol, atol=atol)
        else:
            raise ValueError(f"Unsupported types: {type(tvm_out)}, {type(ort_out)}")

    # Check that number of outputs match.
    assert len(tvm_output) == len(ort_output), "Unequal number of outputs"
    for tvm_out, ort_out in zip(tvm_output, ort_output):
        # TODO Allow configurable tolerance.
        if ort_out is not None:
            _check_output(tvm_out, ort_out)


def run_in_tvm(
    model: ModelProto,
    inputs: dict[str, np.ndarray] | None = None,
    ir_version: int = 8,
    opset: int = 14,
):
    if ir_version is not None:
        model.ir_version = ir_version
    if opset is not None:
        for opset_import in model.opset_import:
            if opset_import.domain in ["", "ai.onnx"]:
                opset_import.version = opset
                break

    inputs = generate_random_inputs(model, inputs)
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    tvm_model, params = relax.frontend.detach_params(tvm_model)

    with tvm.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())

    input_list = [
        inputs[key.name_hint] for key in tvm_model["main"].params if key.name_hint in inputs
    ]
    if params:
        input_list += params["main"]

    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    return vm.get_outputs("main")


@pytest.mark.parametrize(
    "input_names, expected_names",
    [
        ([".", "123"], ["_", "input_123"]),
        ([".", "_"], ["_", "__1"]),
        (["123", "input_123"], ["input_123", "input_123_1"]),
    ],
)
def test_sanitize(input_names, expected_names):
    node = helper.make_node("Add", inputs=input_names, outputs=["output"])
    graph = helper.make_graph(
        [node],
        "test",
        inputs=[
            helper.make_tensor_value_info(str(var), TensorProto.FLOAT, [32, 32])
            for var in input_names
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32]),
        ],
    )
    model = helper.make_model(graph, producer_name="test_sanitizer")

    tvm_model = from_onnx(model)

    for i, param in enumerate(tvm_model["main"].params):
        assert param.name_hint == expected_names[i]


def verify_unary(
    op_name,
    shape,
    attrs={},
    domain=None,
    input_dtype=TensorProto.FLOAT,
    output_dtype=TensorProto.FLOAT,
    opset=14,
):
    test_node = helper.make_node(op_name, ["x"], ["y"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "elemwise_test",
        inputs=[
            helper.make_tensor_value_info("x", input_dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", output_dtype, shape)],
    )

    model = helper.make_model(graph, producer_name="elemwise_test")
    check_correctness(model, opset=opset)


def make_unary_model(
    op_name,
    shape,
    attrs=None,
    domain=None,
    input_dtype=TensorProto.FLOAT,
    output_dtype=TensorProto.FLOAT,
):
    attrs = attrs or {}
    test_node = helper.make_node(op_name, ["x"], ["y"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "elemwise_structural_test",
        inputs=[
            helper.make_tensor_value_info("x", input_dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", output_dtype, shape)],
    )
    return helper.make_model(graph, producer_name="elemwise_structural_test")


def verify_binary(
    op_name, shape_a, shape_b, shape_c, attrs={}, domain=None, dtype=TensorProto.FLOAT, opset=14
):
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "binary_test",
        inputs=[
            helper.make_tensor_value_info("a", dtype, shape_a),
            helper.make_tensor_value_info("b", dtype, shape_b),
        ],
        outputs=[helper.make_tensor_value_info("c", dtype, shape_c)],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    check_correctness(model, opset=opset, check_dtypes=True)


def verify_binary_scalar(op_name, attrs={}, domain=None, dtype=TensorProto.INT32, opset=14):
    a = make_constant_node("a", dtype, [], [4])
    b = make_constant_node("b", dtype, [], [8])
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [a, b, test_node],
        "binary_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("c", dtype, ())],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    model.opset_import[0].version = opset
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

    dtype_str = str(helper.tensor_dtype_to_np_dtype(dtype))
    lhs = np.array(4, dtype=dtype_str)
    rhs = np.array(8, dtype=dtype_str)
    op = {
        "Add": np.add,
        "Sub": np.subtract,
        "Mul": np.multiply,
        "Div": np.divide,
        "Pow": np.power,
        "Mod": np.mod if attrs.get("fmod", 0) else np.fmod,
    }[op_name]
    expected_value = op(lhs, rhs).astype(dtype_str)

    bb = relax.BlockBuilder()
    with bb.function("main", []):
        with bb.dataflow():
            out = bb.emit_output(relax.const(expected_value.item(), dtype_str))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 0)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def verify_compare(op_name, shape, attrs={}, domain=None):
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "compare_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.BOOL, shape)],
    )

    model = helper.make_model(graph, producer_name="compare_test")
    check_correctness(model)


@pytest.mark.parametrize("dynamic", [True, False])
def test_matmul(dynamic):
    matmul_node = helper.make_node("MatMul", ["a", "b"], ["c"])

    a_shape = [32, 48]
    b_shape = [48, 64]
    output_shape = [32, 64]

    if dynamic:
        a_shape = ["?", "?"]

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor(
                "b", TensorProto.FLOAT, b_shape, np.random.normal(size=b_shape).astype("float32")
            )
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, output_shape)],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    inputs = None
    if dynamic:
        inputs = {
            "a": np.random.normal(size=[32, 48]).astype("float32"),
        }
    check_correctness(model, inputs)


@pytest.mark.parametrize(
    ("a_dtype", "b_dtype", "a_shape", "b_shape"),
    [
        (np.int16, np.int16, [2, 3], [3, 4]),
        (np.uint16, np.uint16, [2, 3], [3, 4]),
        (np.int16, np.uint16, [2, 1, 3, 5], [1, 2, 5, 4]),
    ],
)
def test_matmulinteger16(a_dtype, b_dtype, a_shape, b_shape):
    out_dtype = np.uint32 if a_dtype == np.uint16 and b_dtype == np.uint16 else np.int32
    output_shape = [
        *np.broadcast_shapes(tuple(a_shape[:-2]), tuple(b_shape[:-2])),
        a_shape[-2],
        b_shape[-1],
    ]

    node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "matmulinteger16_test",
        inputs=[
            helper.make_tensor_value_info(
                "a", helper.np_dtype_to_tensor_dtype(np.dtype(a_dtype)), a_shape
            ),
            helper.make_tensor_value_info(
                "b", helper.np_dtype_to_tensor_dtype(np.dtype(b_dtype)), b_shape
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "y",
                helper.np_dtype_to_tensor_dtype(np.dtype(out_dtype)),
                output_shape,
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="matmulinteger16_test",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11

    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    a = relax.Var("a", relax.TensorType(a_shape, np.dtype(a_dtype).name))
    b = relax.Var("b", relax.TensorType(b_shape, np.dtype(b_dtype).name))
    bb = relax.BlockBuilder()
    with bb.function("main", [a, b]):
        with bb.dataflow():
            cast_a = bb.emit(relax.op.astype(a, np.dtype(out_dtype).name))
            cast_b = bb.emit(relax.op.astype(b, np.dtype(out_dtype).name))
            out = bb.emit_output(relax.op.matmul(cast_a, cast_b))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_matmulinteger16_ir():
    node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "matmulinteger16_ir_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.UINT16, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.UINT16, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.UINT32, [2, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="matmulinteger16_ir_test",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11

    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((2, 3), dtype="uint16"),
            b: R.Tensor((3, 4), dtype="uint16"),
        ) -> R.Tensor((2, 4), dtype="uint32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="uint32") = R.astype(a, dtype="uint32")
                lv1: R.Tensor((3, 4), dtype="uint32") = R.astype(b, dtype="uint32")
                gv: R.Tensor((2, 4), dtype="uint32") = R.matmul(lv, lv1, out_dtype="void")
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_matmulinteger16_invalid_dtype_raises():
    node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "matmulinteger16_invalid_dtype_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.INT8, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.UINT16, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT32, [2, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="matmulinteger16_invalid_dtype_test",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11

    with pytest.raises(ValueError, match="input A"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_concat():
    verify_binary("Concat", [1, 32], [1, 32], [2, 32], attrs={"axis": 0})


def test_concat_with_param_shape_value():
    """Concat must handle a 1D-int64 initializer mixed with a ShapeExpr when
    keep_params_in_input=True. Standard pattern in PyTorch-exported ONNX
    models for dynamic-batch Reshape: Reshape(x, Concat(Shape(x)[:1], [12]))."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 4])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 12])
    twelve = numpy_helper.from_array(np.array([12], dtype=np.int64), "twelve")
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), "ends")
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Slice", ["x_shape", "starts", "ends"], ["dyn_n"]),
        helper.make_node("Concat", ["dyn_n", "twelve"], ["new_shape"], axis=0),
        helper.make_node("Reshape", ["x", "new_shape"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "concat_param_shape",
        [inp],
        [out],
        initializer=[twelve, starts, ends],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    # Both modes should succeed; previously True crashed with
    # "Op(relax.concat) expects the input to be a Tuple of Tensors".
    from_onnx(model, keep_params_in_input=False)
    from_onnx(model, keep_params_in_input=True)


def test_concat_with_param_tensor_keeps_runtime_param():
    """Concat(input, weight) under keep_params_in_input=True must keep `weight`
    as a runtime param, not fold it into a constant."""
    weight_np = np.arange(8, dtype=np.float32).reshape(2, 4)
    graph = helper.make_graph(
        [helper.make_node("Concat", ["x", "w"], ["y"], axis=0)],
        "concat_param_tensor",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])],
        initializer=[numpy_helper.from_array(weight_np, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    mod, params = relax.frontend.detach_params(from_onnx(model, keep_params_in_input=True))
    assert "w" in [p.name_hint for p in mod["main"].params]
    assert len(params["main"]) == 1
    np.testing.assert_array_equal(params["main"][0].numpy(), weight_np)


@pytest.mark.parametrize("op_name", ["Add", "Sub", "Mul", "Div", "Pow"])
def test_binary(op_name: str):
    verify_binary(op_name, [1, 32], [1, 32], [1, 32])
    verify_binary_scalar(op_name)


def test_div_integer_constant_zero_divisor_raises_valueerror():
    b_init = numpy_helper.from_array(np.array([3, 0, -2, 1], dtype=np.int32), name="b")
    node = helper.make_node("Div", ["a", "b"], ["y"])
    graph = helper.make_graph(
        [node],
        "div_const_zero",
        [helper.make_tensor_value_info("a", TensorProto.INT32, [4])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [4])],
        initializer=[b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9

    with pytest.raises(
        ValueError, match="ONNX Div with integer inputs encountered divisor value 0"
    ):
        from_onnx(model, opset=18, keep_params_in_input=False)


@pytest.mark.parametrize("int_mode", [True, False])
def test_mod(int_mode: bool):
    if int_mode:
        dtype, fmod = TensorProto.INT32, 0
    else:
        dtype, fmod = TensorProto.FLOAT, 1
    verify_binary("Mod", [1, 32], [1, 32], [1, 32], attrs={"fmod": fmod}, dtype=dtype)
    verify_binary_scalar("Mod", attrs={"fmod": fmod}, dtype=dtype)


SHAPE_PARAMS = [
    ([[32, 32], [32, 32]], [32, 32]),
    ([[32, 1], [1, 2]], [32, 2]),
    (
        [
            [
                32,
            ],
            [
                1,
            ],
        ],
        [
            32,
        ],
    ),
    ([[32, 32, 1, 1], [1, 32, 32]], [32, 32, 32, 32]),
    (
        [
            [32, 32, 1, 1],
            [1, 32, 1],
            [
                32,
            ],
        ],
        [32, 32, 32, 32],
    ),
]


@pytest.mark.parametrize("input_shapes, expected_output_shape", SHAPE_PARAMS)
@pytest.mark.parametrize("op_name", ["Min", "Max", "Sum", "Mean"])
def test_multi_input_broadcasting(op_name, input_shapes, expected_output_shape):
    """Multi-input reductions should import broadcast + stack + reduce."""
    num_inputs = len(input_shapes)
    input_names = [f"i{i}" for i in range(num_inputs)]

    input_values_info = []
    for name, shape in zip(input_names, input_shapes):
        input_values_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    test_node = helper.make_node(op_name, input_names, ["output"])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, expected_output_shape)
    graph = helper.make_graph(
        [test_node],
        f"multi_input_{op_name}_test",
        inputs=input_values_info,
        outputs=[output_info],
    )
    model = helper.make_model(graph, producer_name="multi_input_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    inputs = [
        relax.Var(name, relax.TensorType(shape, "float32"))
        for name, shape in zip(input_names, input_shapes)
    ]
    reduce_ops = {
        "Min": relax.op.min,
        "Max": relax.op.max,
        "Sum": relax.op.sum,
        "Mean": relax.op.mean,
    }

    bb = relax.BlockBuilder()
    with bb.function("main", inputs):
        with bb.dataflow():
            broadcasted = [
                bb.emit(relax.op.broadcast_to(inp, relax.ShapeExpr(expected_output_shape)))
                for inp in inputs
            ]
            stacked = bb.emit(relax.op.stack(broadcasted, axis=0))
            out = bb.emit_output(reduce_ops[op_name](stacked, axis=0))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", num_inputs)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("op_name", ["Less", "LessOrEqual", "Greater", "GreaterOrEqual"])
def test_compare(op_name: str):
    verify_compare(op_name, [1, 32])


@pytest.mark.parametrize("op_name", ["And", "Or", "Xor"])
def test_binary_bool(op_name: str):
    verify_binary(op_name, [32, 32], [32, 32], [32, 32], dtype=TensorProto.BOOL)


@pytest.mark.parametrize("op_name", ["BitwiseAnd", "BitwiseOr", "BitwiseXor"])
def test_bitwise(op_name: str):
    verify_binary(op_name, [32, 32], [32, 32], [32, 32], dtype=TensorProto.UINT64, opset=18)


def test_bitwise_not():
    verify_unary(
        "BitwiseNot",
        [32, 32],
        input_dtype=TensorProto.UINT64,
        output_dtype=TensorProto.UINT64,
        opset=18,
    )


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
def test_bitwise_shift(direction: str):
    shape = [32, 32]
    dtype = TensorProto.UINT64
    test_node = helper.make_node("BitShift", ["a", "b"], ["c"], direction=direction)
    graph = helper.make_graph(
        [test_node],
        "binary_test",
        inputs=[
            helper.make_tensor_value_info("a", dtype, shape),
            helper.make_tensor_value_info("b", dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("c", dtype, shape)],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    check_correctness(model, inputs={"b": np.random.randint(0, 8, shape).astype("uint64")})


@pytest.mark.parametrize(
    "op_name",
    [
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Asin",
        "Acos",
        "Atan",
        "Asinh",
        "Acosh",
        "Atanh",
        "Neg",
        "Abs",
        "Log",
        "Exp",
        "Not",
        "Floor",
        "Ceil",
        "Round",
        "IsInf",
        "IsNaN",
        "Sqrt",
        "Relu",
        "Sign",
        "Softplus",
        "Erf",
        "Sigmoid",
        "Softmax",
        "LogSoftmax",
    ],
)
def test_unary(op_name: str):
    input_dtype = TensorProto.FLOAT
    if op_name in [
        "IsNaN",
        "IsInf",
    ]:
        pytest.skip(f"Skipping test {op_name} because current LegalizeOps does not support it.")
    elif op_name == "Not":
        input_dtype = TensorProto.BOOL
        output_dtype = TensorProto.BOOL
    else:
        output_dtype = TensorProto.FLOAT
    verify_unary(op_name, [8, 8, 8], input_dtype=input_dtype, output_dtype=output_dtype)


def test_reciprocal_ir():
    model = make_unary_model("Reciprocal", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.divide(R.const(1.0, "float32"), x)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_identity_ir():
    model = make_unary_model("Identity", [8, 8, 8])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 8, 8), dtype="float32")) -> R.Tensor((8, 8, 8), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((8, 8, 8), dtype="float32") = x
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_elu_ir():
    model = make_unary_model("Elu", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(-1.0, "float32"), lv2)
                lv4: R.Tensor((2, 3), dtype="float32") = R.nn.relu(x)
                gv: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv4)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_hardswish_ir():
    model = make_unary_model("HardSwish", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.add(x, R.const(3.0, "float32"))
                lv1: R.Tensor((2, 3), dtype="float32") = R.clip(
                    lv, R.prim_value(0), R.prim_value(6)
                )
                lv2: R.Tensor((2, 3), dtype="float32") = R.divide(lv1, R.const(6.0, "float32"))
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(x, lv2)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_softsign_ir():
    model = make_unary_model("Softsign", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.abs(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(lv, R.const(1.0, "float32"))
                gv: R.Tensor((2, 3), dtype="float32") = R.divide(x, lv1)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_hardmax_ir():
    model = make_unary_model("Hardmax", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int64") = R.argmax(x, axis=1, keepdims=False)
                gv: R.Tensor((2, 3), dtype="float32") = R.one_hot(
                    lv,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=3,
                    axis=1,
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def make_legacy_softmax_family_axis_expected(op_name: str, input_shape: list[int], axis: int):
    rank = len(input_shape)
    if axis < 0:
        axis += rank

    dim0 = int(np.prod(input_shape[:axis], dtype=np.int64))
    dim1 = int(np.prod(input_shape[axis:], dtype=np.int64))
    flattened_shape = [dim0, dim1]

    x = relax.Var("x", relax.TensorType(input_shape, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            flattened = bb.emit(relax.op.reshape(x, flattened_shape))
            if op_name == "Softmax":
                out = bb.emit(relax.op.nn.softmax(flattened, axis=-1))
            elif op_name == "LogSoftmax":
                out = bb.emit(relax.op.nn.log_softmax(flattened, axis=-1))
            else:
                argmax = bb.emit(relax.op.argmax(flattened, axis=1, keepdims=False))
                out = bb.emit(
                    relax.op.one_hot(
                        argmax,
                        relax.PrimValue(tvm.tirx.const(1.0, "float32")),
                        relax.PrimValue(tvm.tirx.const(0.0, "float32")),
                        dim1,
                        1,
                    )
                )
            reshaped = bb.emit_output(relax.op.reshape(out, input_shape))
        bb.emit_func_output(reshaped)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)
    return expected


def assert_legacy_softmax_family_axis_ir(
    op_name: str, expected_axis: int, axis_attr: int | None = None
):
    attrs = {} if axis_attr is None else {"axis": axis_attr}
    node = helper.make_node(op_name, ["x"], ["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "legacy_softmax_family_axis_ir_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="legacy_softmax_family_axis_ir_test",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
    expected = make_legacy_softmax_family_axis_expected(op_name, [2, 3, 4], expected_axis)
    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_legacy_softmax_family_opset11_default_axis_semantics(op_name: str):
    assert_legacy_softmax_family_axis_ir(op_name, expected_axis=1)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_legacy_softmax_family_opset11_negative_axis_semantics(op_name: str):
    assert_legacy_softmax_family_axis_ir(op_name, expected_axis=-2, axis_attr=-2)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_legacy_softmax_family_opset11_positive_axis_semantics(op_name: str):
    assert_legacy_softmax_family_axis_ir(op_name, expected_axis=0, axis_attr=0)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_legacy_softmax_family_opset11_axis_equals_rank_semantics(op_name: str):
    assert_legacy_softmax_family_axis_ir(op_name, expected_axis=3, axis_attr=3)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax"])
def test_softmax_family_opset13_default_axis_semantics(op_name: str):
    verify_unary(op_name, [2, 3, 4], opset=13)


def test_hardmax_opset13_default_axis_ir():
    model = make_unary_model("Hardmax", [2, 3, 4])
    model.opset_import[0].version = 13
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="int64") = R.argmax(x, axis=2, keepdims=False)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.one_hot(
                    lv,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=4,
                    axis=2,
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_legacy_softmax_family_opset1_ir_semantics(op_name: str):
    node = helper.make_node(op_name, ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "legacy_softmax_family_opset1_ir_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="legacy_softmax_family_opset1_ir_test",
        opset_imports=[helper.make_opsetid("", 1)],
    )
    tvm_model = from_onnx(model, opset=1, keep_params_in_input=True)

    if op_name == "Softmax":

        @I.ir_module
        class ExpectedSoftmax:
            @R.function
            def main(
                x: R.Tensor((2, 3, 4), dtype="float32"),
            ) -> R.Tensor((2, 3, 4), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                    lv1: R.Tensor((2, 12), dtype="float32") = R.nn.softmax(lv, axis=-1)
                    gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedSoftmax)

    elif op_name == "LogSoftmax":

        @I.ir_module
        class ExpectedLogSoftmax:
            @R.function
            def main(
                x: R.Tensor((2, 3, 4), dtype="float32"),
            ) -> R.Tensor((2, 3, 4), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                    lv1: R.Tensor((2, 12), dtype="float32") = R.nn.log_softmax(lv, axis=-1)
                    gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedLogSoftmax)

    else:

        @I.ir_module
        class ExpectedHardmax:
            @R.function
            def main(
                x: R.Tensor((2, 3, 4), dtype="float32"),
            ) -> R.Tensor((2, 3, 4), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                    lv1: R.Tensor((2,), dtype="int64") = R.argmax(lv, axis=1, keepdims=False)
                    lv2: R.Tensor((2, 12), dtype="float32") = R.one_hot(
                        lv1,
                        R.prim_value(T.float32(1.0)),
                        R.prim_value(T.float32(0.0)),
                        depth=12,
                        axis=1,
                    )
                    gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv2, R.shape([2, 3, 4]))
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedHardmax)


def test_round_ties_to_even():
    """ONNX Round must use ties-to-even (banker's rounding), not ties-away-from-zero.

    Per the ONNX spec: "For cases where number is exactly halfway between two
    integers, it rounds to the nearest even integer."
    https://onnx.ai/onnx/operators/onnx__Round.html
    """
    round_node = helper.make_node("Round", ["x"], ["y"])
    graph = helper.make_graph(
        [round_node],
        "round_ties_to_even_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [6])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [6])],
    )
    model = helper.make_model(graph, producer_name="round_ties_to_even_test")
    # Midpoint values: 0.5->0, 1.5->2, 2.5->2, -0.5->0, -1.5->-2, -2.5->-2 (ties-to-even)
    # Ties-away would give: 0.5->1, 1.5->2, 2.5->3, -0.5->-1, -1.5->-2, -2.5->-3
    inputs = {"x": np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype="float32")}
    check_correctness(model, inputs=inputs, opset=11)


@pytest.mark.parametrize("from_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("to_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16])
def test_cast(from_type, to_type):
    cast_node = helper.make_node("Cast", ["a"], ["a_float"], to=to_type)

    graph = helper.make_graph(
        [cast_node],
        "cast_test",
        inputs=[
            helper.make_tensor_value_info("a", from_type, [1, 32]),
        ],
        outputs=[helper.make_tensor_value_info("a_float", to_type, [1, 32])],
    )

    model = helper.make_model(graph, producer_name="cast_test")
    check_correctness(model, opset=13)


@pytest.mark.parametrize("to_type", [TensorProto.INT64, TensorProto.UINT64])
def test_cast_float_to_64bit_int_dynamic(to_type):
    cast_node = helper.make_node("Cast", ["a"], ["b"], to=to_type)
    graph = helper.make_graph(
        [cast_node],
        "cast_float_to_64bit_int_dynamic_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 8])],
        outputs=[helper.make_tensor_value_info("b", to_type, [1, 8])],
    )
    model = helper.make_model(graph, producer_name="cast_float_to_64bit_int_dynamic_test")
    inputs = {"a": np.array([[0.0, 1.2, 2.8, 7.9, 15.1, 31.7, 63.4, 127.9]], dtype=np.float32)}
    check_correctness(model, inputs=inputs, opset=13, check_dtypes=True)


def test_cast_nan_inf_to_int8():
    vals = np.array([300.0, np.nan, np.inf, -np.inf, 50.0, -50.0], dtype=np.float32)
    node = helper.make_node("Cast", inputs=["a"], outputs=["b"], to=TensorProto.INT8)
    graph = helper.make_graph(
        [node],
        "cast_nan_inf_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, list(vals.shape))],
        outputs=[helper.make_tensor_value_info("b", TensorProto.INT8, list(vals.shape))],
    )
    model = helper.make_model(graph, producer_name="cast_nan_inf_test")
    tvm_output = run_in_tvm(model, inputs={"a": vals}, opset=13)
    out_np = tvm_output.numpy()
    expected = np.array([44, 0, 0, 0, 50, -50], dtype=np.int8)
    assert out_np.dtype == np.int8
    np.testing.assert_array_equal(out_np, expected)


def make_gather_expected(data_shape, indices_shape, axis, indices_dtype):
    data = relax.Var("data", relax.TensorType(data_shape, "float32"))
    indices = relax.Var("indices", relax.TensorType(indices_shape, indices_dtype))

    bb = relax.BlockBuilder()
    with bb.function("main", [data, indices]):
        with bb.dataflow():
            data_shape_expr = bb.normalize(relax.op.shape_of(data))
            data_shape_tensor = bb.normalize(relax.op.shape_to_tensor(data_shape_expr))
            axis_extent = bb.normalize(
                relax.op.take(data_shape_tensor, relax.const(axis, "int64"), axis=0, mode="wrap")
            )
            if indices_dtype != "int64":
                axis_extent = bb.normalize(relax.op.astype(axis_extent, indices_dtype))
            normalized_indices = bb.normalize(
                relax.op.where(
                    relax.op.less(indices, relax.const(0, indices_dtype)),
                    relax.op.add(indices, axis_extent),
                    indices,
                )
            )
            out = bb.emit_output(relax.op.take(data, normalized_indices, axis))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)
    return expected


def test_gather():
    def _verify_gather(data_shape, indices, out_shape, axis=0):
        gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=axis)

        if isinstance(indices, list | tuple):
            indices_shape = np.asarray(indices).shape
        else:
            indices_shape = []

        graph = helper.make_graph(
            [gather_node],
            "gather_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(
            graph, producer_name="gather_test", opset_imports=[helper.make_opsetid("", 14)]
        )
        tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(
            tvm_model, make_gather_expected(data_shape, list(indices_shape), axis, "int64")
        )

    _verify_gather([5, 4, 3, 2], [0, 1, 3], [3, 4, 3, 2])
    _verify_gather([3], 0, [])
    _verify_gather([3, 3], [[0, 2]], [3, 1, 2], 1)


@pytest.mark.parametrize(
    "axis, indices, out_shape",
    [
        (0, [-1, 0], [2, 4]),
        (1, [-1, 0], [3, 2]),
        (
            1,
            [[-1, 0], [1, -2]],
            [3, 2, 2],
        ),
    ],
)
@pytest.mark.parametrize("indices_type", [TensorProto.INT64, TensorProto.INT32])
def test_gather_negative_indices(axis, indices, out_shape, indices_type):
    gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=axis)
    indices_shape = np.asarray(indices).shape

    graph = helper.make_graph(
        [gather_node],
        "gather_negative_indices_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("indices", indices_type, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(
        graph,
        producer_name="gather_negative_indices_test",
        opset_imports=[helper.make_opsetid("", 14)],
    )
    indices_np_dtype = {
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
    }[indices_type]
    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
    tvm.ir.assert_structural_equal(
        tvm_model,
        make_gather_expected(
            [3, 4],
            list(indices_shape),
            axis,
            np.dtype(indices_np_dtype).name,
        ),
    )


@pytest.mark.parametrize("indices_type", [TensorProto.INT64, TensorProto.INT32])
def test_gather_negative_indices_ir_normalization(indices_type):
    gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=1)
    graph = helper.make_graph(
        [gather_node],
        "gather_negative_indices_ir_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("indices", indices_type, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
    )

    model = helper.make_model(graph, producer_name="gather_negative_indices_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    if indices_type == TensorProto.INT32:

        @I.ir_module
        class ExpectedInt32:
            @R.function
            def main(
                data: R.Tensor((3, 4), dtype="float32"),
                indices: R.Tensor((2,), dtype="int32"),
            ) -> R.Tensor((3, 2), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Shape([3, 4]) = R.shape_of(data)
                    lv1: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                    lv2: R.Tensor((), dtype="int64") = R.take(
                        lv1, R.const(1, "int64"), axis=0, mode="wrap"
                    )
                    lv3: R.Tensor((2,), dtype="bool") = R.less(indices, R.const(0, "int32"))
                    lv4: R.Tensor((), dtype="int32") = R.astype(lv2, dtype="int32")
                    lv5: R.Tensor((2,), dtype="int32") = R.add(indices, lv4)
                    lv6: R.Tensor((2,), dtype="int32") = R.where(lv3, lv5, indices)
                    gv: R.Tensor((3, 2), dtype="float32") = R.take(data, lv6, axis=1, mode="fast")
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedInt32)

    else:

        @I.ir_module
        class ExpectedInt64:
            @R.function
            def main(
                data: R.Tensor((3, 4), dtype="float32"),
                indices: R.Tensor((2,), dtype="int64"),
            ) -> R.Tensor((3, 2), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Shape([3, 4]) = R.shape_of(data)
                    lv1: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                    lv2: R.Tensor((2,), dtype="bool") = R.less(indices, R.const(0, "int64"))
                    lv3: R.Tensor((), dtype="int64") = R.take(
                        lv1, R.const(1, "int64"), axis=0, mode="wrap"
                    )
                    lv4: R.Tensor((2,), dtype="int64") = R.add(indices, lv3)
                    lv5: R.Tensor((2,), dtype="int64") = R.where(lv2, lv4, indices)
                    gv: R.Tensor((3, 2), dtype="float32") = R.take(data, lv5, axis=1, mode="fast")
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedInt64)


@pytest.mark.parametrize(
    "data_shape, indices_shape, axis",
    [
        ([3, 4, 5], [1, 4, 5], 0),
        ([3, 4, 5], [3, 2, 5], 1),
        ([3, 4, 5], [3, 4, 2], 2),
    ],
)
def test_gather_elements(data_shape, indices_shape, axis):
    gather_elements_node = helper.make_node("GatherElements", ["data", "indices"], ["y"], axis=axis)

    graph = helper.make_graph(
        [gather_elements_node],
        "gather_elements_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, indices_shape)],
    )

    model = helper.make_model(graph, producer_name="gather_elements_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(0, data_shape[axis], indices_shape).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize(
    "data_shape, indices_shape, batch_dims",
    [
        ([2, 2], [2, 2], 0),
        ([2, 2], [2, 1], 0),
        ([2, 2, 2], [1], 0),
        ([2, 2, 2], [2, 2], 0),
        ([2, 2, 2], [2, 1, 2], 0),
        ([2, 2, 2], [2, 2], 1),
        ([2, 2, 2], [2, 1], 1),
    ],
)
def test_gather_nd(data_shape, indices_shape, batch_dims):
    gather_nd_node = helper.make_node("GatherND", ["data", "indices"], ["y"], batch_dims=batch_dims)

    graph = helper.make_graph(
        [gather_nd_node],
        "gather_nd_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )

    model = helper.make_model(graph, producer_name="gather_nd_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(0, 2, indices_shape).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize(("name", "opset"), [("Scatter", 10), ("ScatterElements", 11)])
def test_scatter(axis: int, name: str, opset: int):
    if axis != 1:
        pytest.skip("The current topi impl is wrong, which only works for axis=1")
    input_shape = [16, 16, 16]
    indices_shape = [8, 8, 8]
    updates_shape = [8, 8, 8]
    output_shape = [16, 16, 16]
    node = helper.make_node(name, ["data", "indices", "updates"], ["output"], axis=axis)
    graph = helper.make_graph(
        [node],
        "scatter_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="scatter_test")
    indices = np.random.randint(0, 16, indices_shape)
    check_correctness(model, inputs={"indices": indices}, opset=opset)


@pytest.mark.parametrize(
    "reduction, opset, data, indices, updates",
    [
        (
            None,
            11,
            np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"),
            np.array([[2, 0, 1], [1, 2, 0]], dtype="int64"),
            np.array([[30, 10, 20], [50, 60, 40]], dtype="float32"),
        ),
        (
            "none",
            18,
            np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"),
            np.array([[2, 0, 1], [1, 2, 0]], dtype="int64"),
            np.array([[30, 10, 20], [50, 60, 40]], dtype="float32"),
        ),
        (
            "add",
            16,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "mul",
            16,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "min",
            18,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "max",
            18,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
    ],
)
def test_scatter_elements_reduction(reduction, opset, data, indices, updates):
    attrs = {"axis": 1}
    if reduction is not None:
        attrs["reduction"] = reduction
    scatter_elements_node = helper.make_node(
        "ScatterElements", ["data", "indices", "updates"], ["output"], **attrs
    )

    graph = helper.make_graph(
        [scatter_elements_node],
        "scatter_elements_reduction_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, list(updates.shape)),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(data.shape))],
    )
    model = helper.make_model(graph, producer_name="scatter_elements_reduction_test")

    check_correctness(
        model,
        inputs={"data": data, "indices": indices, "updates": updates},
        opset=opset,
    )


def test_scatter_elements_invalid_reduction():
    data_shape = [2, 3]
    scatter_elements_node = helper.make_node(
        "ScatterElements",
        ["data", "indices", "updates"],
        ["output"],
        axis=1,
        reduction="unsupported",
    )

    graph = helper.make_graph(
        [scatter_elements_node],
        "scatter_elements_invalid_reduction_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, data_shape),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, data_shape),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
    )
    model = helper.make_model(graph, producer_name="scatter_elements_invalid_reduction_test")

    with pytest.raises(ValueError, match="Only .* reductions are supported, but got unsupported"):
        from_onnx(model, opset=18, keep_params_in_input=True)


@pytest.mark.parametrize("reduction", ["none", "add", "mul"])
def test_scatter_nd(reduction):
    def verify_scatter_nd(data_shape, indices_shape, updates_shape):
        scatter_nd_node = helper.make_node(
            "ScatterND",
            ["data", "indices", "updates"],
            ["output"],
            reduction=reduction,
        )

        graph = helper.make_graph(
            [scatter_nd_node],
            "scatter_nd_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
            ],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
        )

        model = helper.make_model(graph, producer_name="scatter_nd_test")

        indices = np.random.choice(data_shape[0], indices_shape)
        check_correctness(model, inputs={"indices": indices}, opset=16)

    verify_scatter_nd([8], [4, 1], [4])
    verify_scatter_nd([4, 4, 4], [2, 1], [2, 4, 4])
    verify_scatter_nd([4, 5, 6], [2, 3, 2], [2, 3, 6])
    verify_scatter_nd([10], [5, 1], [5])


def make_compress_expected(tensor_shape: list[int], condition_shape: list[int], axis: int | None):
    num_nonzero = tvm.tirx.Var("num_nonzero", "int64")
    tensor = relax.Var("tensor", relax.TensorType(tensor_shape, "float32"))
    condition = relax.Var("condition", relax.TensorType(condition_shape, "bool"))

    bb = relax.BlockBuilder()
    with bb.function("main", [tensor, condition]):
        with bb.dataflow():
            nonzero_indices = bb.match_cast(
                relax.op.nonzero(condition),
                relax.TensorType([1, num_nonzero], "int64"),
            )
            if axis is None:
                flattened = bb.emit(relax.op.reshape(tensor, [int(np.prod(tensor_shape))]))
                indices = bb.emit(relax.op.reshape(nonzero_indices, [num_nonzero]))
                out = bb.emit_output(relax.op.take(flattened, indices, axis=0))
            else:
                indices = bb.emit(relax.op.reshape(nonzero_indices, [num_nonzero]))
                out = bb.emit_output(relax.op.take(tensor, indices, axis=axis))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)
    return expected


@pytest.mark.parametrize("tensor_shape", [[32, 32]])
@pytest.mark.parametrize("condition_shape", [None, [8], [16]])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_compress(
    tensor_shape: list[int],
    condition_shape: list[int] | None,
    axis: int | None,
):
    if condition_shape is None and axis is None:
        pytest.skip("Either condition_shape or axis must be specified")
    if condition_shape is None:
        condition_shape = [tensor_shape[axis]]
    compress_node = helper.make_node("Compress", ["tensor", "condition"], ["output"], axis=axis)
    graph = helper.make_graph(
        [compress_node],
        "compress_test",
        inputs=[
            helper.make_tensor_value_info("tensor", TensorProto.FLOAT, tensor_shape),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, condition_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [])
        ],  # shape is unknown
    )
    model = helper.make_model(graph, producer_name="compress_test")
    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
    tvm.ir.assert_structural_equal(
        tvm_model, make_compress_expected(tensor_shape, condition_shape, axis)
    )


def test_size():
    test_node = helper.make_node("Size", ["x"], ["y"])
    input_shape = [3, 3, 3]
    graph = helper.make_graph(
        [test_node],
        "size_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, [3])],
    )

    model = helper.make_model(graph, producer_name="size_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    data = relax.Var("x", relax.TensorType(input_shape, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow():
            out = bb.emit_output(relax.op.size(data))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_eye_like(k: int):
    node = helper.make_node("EyeLike", ["x"], ["y"], k=k)
    graph = helper.make_graph(
        [node],
        "eye_like_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="eye_like_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    x = relax.Var("x", relax.TensorType([32, 32], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_output(relax.op.eye_like(x, k, "float32"))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("alpha", [None, 0.25, 1.0])
@pytest.mark.parametrize("beta", [None, 0.35, 1.0])
@pytest.mark.parametrize("useC", [False, True])
def test_gemm(alpha, beta, useC):
    if useC:
        gemm_node = helper.make_node(
            "Gemm", ["a", "b", "c"], ["y"], alpha=alpha, beta=beta, transA=1, transB=1
        )
    else:
        gemm_node = helper.make_node(
            "Gemm", ["a", "b"], ["y"], alpha=alpha, beta=beta, transA=1, transB=1
        )

    inputs = [
        helper.make_tensor_value_info("a", TensorProto.FLOAT, [4, 3]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [5, 4]),
    ]
    if useC:
        inputs.append(helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 5]))

    graph = helper.make_graph(
        [gemm_node],
        "gemm_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5])],
    )

    model = helper.make_model(graph, producer_name="gemm_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    a = relax.Var("a", relax.TensorType([4, 3], "float32"))
    b = relax.Var("b", relax.TensorType([5, 4], "float32"))
    params = [a, b]
    if useC:
        c = relax.Var("c", relax.TensorType([1, 5], "float32"))
        params.append(c)
    else:
        c = None

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            gemm_a = a
            if alpha is not None and alpha != 1.0:
                gemm_a = bb.emit(relax.op.multiply(gemm_a, relax.const(alpha, "float32")))
            gemm_a = bb.emit(relax.op.permute_dims(gemm_a, [1, 0]))
            gemm_b = bb.emit(relax.op.permute_dims(b, [1, 0]))
            if c is not None:
                y = bb.emit(relax.op.matmul(gemm_a, gemm_b))
                gemm_c = c
                if beta is not None and beta != 1.0:
                    gemm_c = bb.emit(relax.op.multiply(gemm_c, relax.const(beta, "float32")))
                gv = bb.emit_output(relax.op.add(y, gemm_c))
            else:
                gv = bb.emit_output(relax.op.matmul(gemm_a, gemm_b))
        bb.emit_func_output(gv)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", len(params))

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize(
    "in_shape, shape, out_shape",
    [
        ([7, 32, 32, 8], [224, 256], [224, 256]),
        ([7, 32, 32, 8], [-1, 8192], [7, 8192]),
        ([7, 32, 32, 8], [0, 32, 32, 8], [7, 32, 32, 8]),
    ],
)
def test_reshape(in_shape, shape, out_shape):
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"])

    graph = helper.make_graph(
        [reshape_node],
        "reshape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape),
        ],
        initializer=[helper.make_tensor("shape", TensorProto.INT64, [len(shape)], shape)],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="reshape_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    data = relax.Var("data", relax.TensorType(in_shape, "float32"))
    shape_var = relax.Var("shape", relax.TensorType([len(shape)], "int64"))
    bb = relax.BlockBuilder()
    with bb.function("main", [data, shape_var]):
        with bb.dataflow():
            out = bb.emit_output(relax.op.reshape(data, out_shape))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize(
    "target_shape, output_shape",
    [
        ([-1], [3]),
        ([1, 3], [1, 3]),
        ([3, 1], [3, 1]),
    ],
)
def test_reshape_shape_output(target_shape, output_shape):
    shape_node = helper.make_node("Shape", ["data"], ["shape_out"])
    reshape_node = helper.make_node("Reshape", ["shape_out", "target_shape"], ["reshaped"])

    data_shape = [2, 3, 4]

    graph = helper.make_graph(
        [shape_node, reshape_node],
        "reshape_shape_output",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
        ],
        initializer=[
            helper.make_tensor("target_shape", TensorProto.INT64, [len(target_shape)], target_shape)
        ],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.INT64, output_shape)],
    )
    model = helper.make_model(graph, producer_name="reshape_shape_output")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    if target_shape == [-1]:

        @I.ir_module
        class ExpectedFlattenShape:
            @R.function
            def main(
                data: R.Tensor((2, 3, 4), dtype="float32"),
                target_shape: R.Tensor((1,), dtype="int64"),
            ) -> R.Shape([2, 3, 4]):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Shape([2, 3, 4]) = R.shape([2, 3, 4])
                    R.output(gv)
                return gv

        expected = ExpectedFlattenShape

    elif target_shape == [1, 3]:

        @I.ir_module
        class ExpectedRank2Shape:
            @R.function
            def main(
                data: R.Tensor((2, 3, 4), dtype="float32"),
                target_shape: R.Tensor((2,), dtype="int64"),
            ) -> R.Tensor((1, 3), dtype="int64"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((3,), dtype="int64") = R.shape_to_tensor(R.shape([2, 3, 4]))
                    gv: R.Tensor((1, 3), dtype="int64") = R.reshape(lv, R.shape([1, 3]))
                    R.output(gv)
                return gv

        expected = ExpectedRank2Shape

    else:

        @I.ir_module
        class ExpectedRank2ColumnShape:
            @R.function
            def main(
                data: R.Tensor((2, 3, 4), dtype="float32"),
                target_shape: R.Tensor((2,), dtype="int64"),
            ) -> R.Tensor((3, 1), dtype="int64"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((3,), dtype="int64") = R.shape_to_tensor(R.shape([2, 3, 4]))
                    gv: R.Tensor((3, 1), dtype="int64") = R.reshape(lv, R.shape([3, 1]))
                    R.output(gv)
                return gv

        expected = ExpectedRank2ColumnShape

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_transpose():
    node = helper.make_node("Transpose", ["x"], ["y"], perm=[1, 2, 0])
    graph = helper.make_graph(
        [node],
        "transpose_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [32, 32, 32])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32, 32])],
    )
    model = helper.make_model(graph, producer_name="transpose_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((32, 32, 32), dtype="float32")) -> R.Tensor(
            (32, 32, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((32, 32, 32), dtype="float32") = R.permute_dims(x, axes=[1, 2, 0])
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_transpose_scalar():
    """Test Transpose with scalar inputs - should return scalar unchanged."""
    scalar_node = helper.make_node("Transpose", ["x"], ["y"])
    graph = helper.make_graph(
        [scalar_node],
        "transpose_scalar_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="transpose_scalar_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedScalar:
        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = x
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedScalar)

    scalar_constant = helper.make_node(
        "Constant",
        [],
        ["scalar"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [5.0]),
    )

    transpose_node = helper.make_node("Transpose", ["scalar"], ["y"])
    graph = helper.make_graph(
        [scalar_constant, transpose_node],
        "transpose_scalar_constant_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="transpose_scalar_constant_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedConstant:
        @R.function
        def main() -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.const(5.0, "float32")
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedConstant)


def test_transpose_axes_validation():
    """Test Transpose validation - perm axes count must match tensor dimensions"""

    def assert_transpose_ir(input_shape, axes, output_shape, name):
        transpose_node = helper.make_node("Transpose", ["x"], ["y"], perm=axes)
        graph = helper.make_graph(
            [transpose_node],
            name,
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(graph, producer_name=name)
        tvm_model = from_onnx(model, keep_params_in_input=True)

        x = relax.Var("x", relax.TensorType(input_shape, "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit_output(relax.op.permute_dims(x, axes=axes))
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 1)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    # Test 1D tensor with correct perm
    assert_transpose_ir([10], [0], [10], "transpose_1d_valid_test")

    # Test 2D tensor with correct perm
    assert_transpose_ir([3, 4], [1, 0], [4, 3], "transpose_2d_valid_test")

    # Test 3D tensor with correct perm
    assert_transpose_ir([2, 3, 4], [2, 0, 1], [4, 2, 3], "transpose_3d_valid_test")


def assert_static_unsqueeze_ir(
    model: ModelProto,
    input_shape: list[int],
    axes: list[int],
    *,
    opset: int,
    axes_as_param: bool,
):
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    if axes_as_param:
        tvm_model["main"] = tvm_model["main"].without_attr("params")

    data = relax.Var("a", relax.TensorType(input_shape, "float32"))
    params = [data]
    if axes_as_param:
        params.append(relax.Var("axes", relax.TensorType([len(axes)], "int64")))

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            out = data
            sorted_axes = sorted(axes)
            for index, axis in enumerate(sorted_axes):
                expanded = relax.op.expand_dims(out, axis=axis)
                if index == len(sorted_axes) - 1:
                    out = bb.emit_output(expanded)
                else:
                    out = bb.emit(expanded)
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_unsqueeze():
    axes = [0, 2, 3]
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])
    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [3], vals=axes)],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(
        graph, producer_name="unsqueeze_test", opset_imports=[helper.make_opsetid("", 13)]
    )
    assert_static_unsqueeze_ir(
        model,
        [32, 32],
        axes,
        opset=13,
        axes_as_param=True,
    )


def test_unsqueeze_scalar_input():
    axes = [0, 1]
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_scalar_input",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [2], vals=axes)],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1])],
    )

    model = helper.make_model(
        graph,
        producer_name="unsqueeze_scalar_input_test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    assert_static_unsqueeze_ir(model, [], axes, opset=13, axes_as_param=True)


def test_unsqueeze_dynamic_axes_ir():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_dynamic_axes_ir",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 32, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_dynamic_axes_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((32, 32), dtype="float32"),
            axes: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=4):
            R.func_attr({"num_input": 2})
            unsqueeze_dim_0 = T.int64()
            unsqueeze_dim_1 = T.int64()
            unsqueeze_dim_2 = T.int64()
            unsqueeze_dim_3 = T.int64()
            with R.dataflow():
                lv: R.Shape([32, 32]) = R.shape_of(a)
                lv1: R.Tensor((2,), dtype="bool") = R.less(axes, R.const(0, "int64"))
                lv2: R.Tensor((2,), dtype="int64") = R.add(axes, R.const(4, "int64"))
                lv3: R.Tensor((4,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(4), R.prim_value(1), dtype="int64"
                )
                lv4: R.Tensor((2,), dtype="int64") = R.where(lv1, lv2, axes)
                lv5: R.Tensor((4, 1), dtype="int64") = R.expand_dims(lv3, axis=[1])
                lv6: R.Tensor((1, 2), dtype="int64") = R.expand_dims(lv4, axis=[0])
                lv7: R.Tensor((4, 2), dtype="bool") = R.equal(lv5, lv6)
                lv8: R.Tensor((4, 2), dtype="int64") = R.astype(lv7, dtype="int64")
                lv9: R.Tensor((4,), dtype="int64") = R.sum(lv8, axis=[1], keepdims=False)
                lv10: R.Tensor((4,), dtype="int64") = R.subtract(R.const(1, "int64"), lv9)
                lv11: R.Tensor((4,), dtype="int64") = R.cumsum(
                    lv10, axis=0, dtype="void", exclusive=False
                )
                lv12: R.Tensor((4,), dtype="int64") = R.subtract(lv11, R.const(1, "int64"))
                lv13: R.Tensor((4,), dtype="bool") = R.less(lv12, R.const(0, "int64"))
                lv14: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                lv15: R.Tensor((4,), dtype="int64") = R.where(lv13, R.const(0, "int64"), lv12)
                lv16: R.Tensor((4,), dtype="bool") = R.greater(lv9, R.const(0, "int64"))
                lv17: R.Tensor((4,), dtype="int64") = R.take(lv14, lv15, axis=0, mode="fast")
                lv18: R.Tensor((4,), dtype="int64") = R.match_cast(
                    R.where(lv16, R.const(1, "int64"), lv17), R.Tensor((4,), dtype="int64")
                )
                lv19: R.Shape(ndim=4) = R.tensor_to_shape(lv18)
                lv20: R.Shape(
                    [unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3]
                ) = R.match_cast(
                    lv19,
                    R.Shape([unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3]),
                )
                gv: R.Tensor(
                    (unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3),
                    dtype="float32",
                ) = R.reshape(
                    a,
                    R.shape([unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3]),
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_unsqueeze_dynamic_axes_rank_validation():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_dynamic_axes_rank_validation",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1, 2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 32, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_dynamic_axes_rank_validation_test")
    with pytest.raises(ValueError, match="Expected a 1-D tensor"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_unsqueeze_duplicate_axes_validation():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_duplicate_axes_validation",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [2], vals=[0, 0])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 32, 32])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_duplicate_axes_validation_test")
    with pytest.raises(ValueError, match="axes must be unique"):
        from_onnx(model, opset=13)


def test_unsqueeze_v1():
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Unsqueeze-1
    axes = [0, 2, 3]
    unsqueeze_node = helper.make_node("Unsqueeze", ["a"], ["b"], axes=axes)
    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_v1",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(
        graph, producer_name="unsqueeze_v1_test", opset_imports=[helper.make_opsetid("", 6)]
    )
    assert_static_unsqueeze_ir(
        model,
        [32, 32],
        axes,
        opset=10,
        axes_as_param=False,
    )


def test_gelu():
    verify_unary("Gelu", [32, 32], domain="com.microsoft")


def test_gelu_approximate():
    """Test Gelu with approximate attribute from ONNX Opset 20."""
    # Test Gelu with approximate="tanh"
    verify_unary("Gelu", [32, 32], attrs={"approximate": "tanh"}, opset=20)
    # Test Gelu with approximate="none" (default, same as standard Gelu)
    verify_unary("Gelu", [32, 32], attrs={"approximate": "none"}, opset=20)


def test_bias_gelu():
    bias_gelu_node = helper.make_node("BiasGelu", ["a", "b"], ["c"], domain="com.microsoft")
    graph = helper.make_graph(
        [bias_gelu_node],
        "bias_gelu_structural_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, producer_name="bias_gelu_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((2, 3), dtype="float32"),
            b: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.add(a, b)
                gv: R.Tensor((2, 3), dtype="float32") = R.nn.gelu(lv)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_fast_gelu():
    """Test FastGelu with and without bias"""
    fast_gelu_node = helper.make_node("FastGelu", ["x"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [fast_gelu_node],
        "fast_gelu_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, producer_name="fast_gelu_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(0.5, "float32"), x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.79788458347320557, "float32"), x
                )
                lv2: R.Tensor((2, 3), dtype="float32") = R.multiply(x, x)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(lv2, x)
                lv4: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.035677406936883926, "float32"), lv3
                )
                lv5: R.Tensor((2, 3), dtype="float32") = R.add(lv1, lv4)
                lv6: R.Tensor((2, 3), dtype="float32") = R.tanh(lv5)
                lv7: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1.0, "float32"), lv6)
                lv8: R.Tensor((2, 3), dtype="float32") = R.multiply(lv, lv7)
                gv: R.Tensor((2, 3), dtype="float32") = lv8
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)

    fast_gelu_with_bias_node = helper.make_node(
        "FastGelu", ["x", "bias"], ["y"], domain="com.microsoft"
    )
    graph_with_bias = helper.make_graph(
        [fast_gelu_with_bias_node],
        "fast_gelu_with_bias_structural_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model_with_bias = helper.make_model(
        graph_with_bias, producer_name="fast_gelu_with_bias_structural_test"
    )
    tvm_model_with_bias = from_onnx(model_with_bias, keep_params_in_input=True)

    @I.ir_module
    class ExpectedWithBias:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            bias: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.add(x, bias)
                lv1: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(0.5, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.79788458347320557, "float32"), lv
                )
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(lv, lv)
                lv4: R.Tensor((2, 3), dtype="float32") = R.multiply(lv3, lv)
                lv5: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.035677406936883926, "float32"), lv4
                )
                lv6: R.Tensor((2, 3), dtype="float32") = R.add(lv2, lv5)
                lv7: R.Tensor((2, 3), dtype="float32") = R.tanh(lv6)
                lv8: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1.0, "float32"), lv7)
                lv9: R.Tensor((2, 3), dtype="float32") = R.multiply(lv1, lv8)
                gv: R.Tensor((2, 3), dtype="float32") = lv9
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model_with_bias, ExpectedWithBias)


def test_where():
    where_node = helper.make_node("Where", ["a", "b", "c"], ["d"])

    graph = helper.make_graph(
        [where_node],
        "where_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.BOOL, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("d", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="where_test")
    check_correctness(model)


@pytest.mark.parametrize("min", [True, False])
@pytest.mark.parametrize("max", [True, False])
def test_clip(min, max):
    if min and max:
        clip_node = helper.make_node("Clip", ["input", "min", "max"], ["output"])
    elif min:
        clip_node = helper.make_node("Clip", ["input", "min"], ["output"])
    elif max:
        clip_node = helper.make_node("Clip", ["input", "max"], ["output"])
    else:
        clip_node = helper.make_node("Clip", ["input"], ["output"])

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 64])]
    if min:
        inputs.append(helper.make_tensor_value_info("min", TensorProto.FLOAT, ()))
    if max:
        inputs.append(helper.make_tensor_value_info("max", TensorProto.FLOAT, ()))

    graph = helper.make_graph(
        [clip_node],
        "clip_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 64])],
    )

    model = helper.make_model(graph, producer_name="clip_test")
    model.opset_import[0].version = 14
    tvm_model = from_onnx(model, keep_params_in_input=True)

    data = relax.Var("input", relax.TensorType([32, 64], "float32"))
    params = [data]
    if min:
        min_bound = relax.Var("min", relax.TensorType([], "float32"))
        params.append(min_bound)
    elif max:
        # The existing max-only case passes the bound as the second ONNX input,
        # which ONNX defines as the min bound.
        min_bound = relax.Var("max", relax.TensorType([], "float32"))
        params.append(min_bound)
    else:
        min_bound = None

    if max and min:
        max_bound = relax.Var("max", relax.TensorType([], "float32"))
        params.append(max_bound)
    else:
        max_bound = None

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            out = data
            if min_bound is not None:
                lo = bb.emit(
                    relax.op.where(
                        relax.op.isnan(min_bound),
                        relax.const(float("-inf"), "float32"),
                        min_bound,
                    )
                )
                out = bb.emit_te(topi.maximum, out, lo)
            if max_bound is not None:
                hi = bb.emit(
                    relax.op.where(
                        relax.op.isnan(max_bound),
                        relax.const(float("inf"), "float32"),
                        max_bound,
                    )
                )
                out = bb.emit_te(topi.minimum, out, hi)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", len(params))

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("min", [-6.0, 0.0])
@pytest.mark.parametrize("max", [6.0])
def test_clip_v6(max, min):
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-6
    clip_node = helper.make_node("Clip", ["input"], ["output"], max=max, min=min)
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 64])]
    graph = helper.make_graph(
        [clip_node],
        "clip_v6_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 64])],
    )
    model = helper.make_model(
        graph, producer_name="clip_v6_test", opset_imports=[helper.make_opsetid("", 6)]
    )
    tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)

    data = relax.Var("input", relax.TensorType([32, 64], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow():
            out = bb.emit_te(topi.maximum, data, min)
            out = bb.emit_te(topi.minimum, out, max)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize(
    "min,max",
    [
        pytest.param(
            np.array(0.0, dtype=np.float32),
            np.array(6.0, dtype=np.float32),
        ),
        pytest.param(
            np.array(0.0, dtype=np.float32),
            np.array(np.nan, dtype=np.float32),
        ),
        pytest.param(
            np.array(np.nan, dtype=np.float32),
            np.array(6.0, dtype=np.float32),
        ),
        pytest.param(
            np.array(np.nan, dtype=np.float32),
            np.array(np.nan, dtype=np.float32),
        ),
    ],
)
@pytest.mark.parametrize(
    "input",
    [
        np.array([0.5, -3.0, 4.5, 11.0, 7.0], dtype=np.float32),
    ],
)
def test_clip_v13(input, min, max):
    # Opset 13: tensor min/max. NaN bound => unbounded on that side (ORT).
    clip_node = helper.make_node("Clip", ["input", "min", "max"], ["output"])
    graph = helper.make_graph(
        [clip_node],
        "clip_v13_nan_max",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("min", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("max", TensorProto.FLOAT, []),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [5])],
    )
    model = helper.make_model(graph, producer_name="clip_v13_nan_max")
    check_correctness(
        model,
        inputs={"input": input, "min": min, "max": max},
        opset=13,
    )


def test_equal():
    equal_node = helper.make_node("Equal", ["a", "b"], ["output"])

    graph = helper.make_graph(
        [equal_node],
        "equal_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="equal_test")
    check_correctness(
        model, {"a": np.zeros([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )
    check_correctness(
        model, {"a": np.ones([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )
    check_correctness(model)


def test_shape():
    shape_node = helper.make_node("Shape", ["data"], ["output"])

    graph = helper.make_graph(
        [shape_node],
        "shape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4, 5, 6]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [4])],
    )

    model = helper.make_model(graph, producer_name="shape_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((3, 4, 5, 6), dtype="float32")) -> R.Shape([3, 4, 5, 6]):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([3, 4, 5, 6]) = R.shape([3, 4, 5, 6])
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("upper", [True, False])
def test_trilu(upper: bool):
    node = helper.make_node("Trilu", ["x"], ["y"], upper=upper)
    graph = helper.make_graph(
        [node],
        "trilu_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 5, 5])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5, 5])],
    )
    model = helper.make_model(graph, producer_name="trilu_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    x = relax.Var("x", relax.TensorType([3, 5, 5], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            trilu = relax.op.triu if upper else relax.op.tril
            out = bb.emit_output(trilu(x, 0))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("k_value", [-1, 0, 1])
def test_trilu_with_const_k(k_value: int):
    """test_trilu_with_const_k"""

    input_shape = [2, 3, 3]

    graph = helper.make_graph(
        [
            make_constant_node("k", onnx.TensorProto.INT64, [1], [k_value]),
            helper.make_node("Trilu", inputs=["x", "k"], outputs=["y"]),
        ],
        "trilu_graph",
        inputs=[
            helper.make_tensor_value_info("x", onnx.TensorProto.DOUBLE, input_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="trilu_graph")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    x = relax.Var("x", relax.TensorType(input_shape, "float64"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_output(relax.op.triu(x, k_value))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_selu():
    model = make_unary_model("Selu", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(-1.6732631921768188, "float32"), lv2
                )
                lv4: R.Tensor((2, 3), dtype="float32") = R.nn.relu(x)
                lv5: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv4)
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(1.0507010221481323, "float32"), lv5
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("Selu", [2, 3], attrs={"alpha": 0.25, "gamma": 0.3})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(-0.25, "float32"), lv2)
                lv4: R.Tensor((2, 3), dtype="float32") = R.nn.relu(x)
                lv5: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv4)
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.30000001192092896, "float32"), lv5
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedCustom)


def test_mish():
    model = make_unary_model("Mish", [2, 3])
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.log(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.tanh(lv2)
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(x, lv3)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_prelu():
    def _assert_prelu_ir(slope_shape):
        prelu_node = helper.make_node("PRelu", ["a", "b"], ["c"])
        graph = helper.make_graph(
            [prelu_node],
            "prelu_structural_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, [3, 32, 32]),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, slope_shape),
            ],
            outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [3, 32, 32])],
        )
        model = helper.make_model(graph, producer_name="prelu_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)

        data = relax.Var("a", relax.TensorType([3, 32, 32], "float32"))
        slope = relax.Var("b", relax.TensorType(slope_shape, "float32"))
        if all(dim == 1 for dim in slope_shape) or len(slope_shape) == 1:
            slope_axis = len(slope_shape) - 1
            prelu_axis = 2
        else:
            non_one_axes = [axis for axis, dim in enumerate(slope_shape) if dim != 1]
            assert len(non_one_axes) == 1
            slope_axis = non_one_axes[0]
            prelu_axis = slope_axis

        bb = relax.BlockBuilder()
        with bb.function("main", [data, slope]):
            with bb.dataflow():
                flattened_slope = bb.emit(relax.op.reshape(slope, [slope_shape[slope_axis]]))
                out = bb.emit_output(relax.op.nn.prelu(data, flattened_slope, prelu_axis))
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 2)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    _assert_prelu_ir([1])
    _assert_prelu_ir([1, 1])
    _assert_prelu_ir([32])
    _assert_prelu_ir([3, 1, 1])


def test_thresholded_relu():
    model = make_unary_model("ThresholdedRelu", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(x, R.const(1.0, "float32"))
                lv1: R.Tensor((2, 3), dtype="float32") = R.astype(lv, dtype="float32")
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(lv1, x)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("ThresholdedRelu", [2, 3], attrs={"alpha": -0.01})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(
                    x, R.const(-0.0099999997764825821, "float32")
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.astype(lv, dtype="float32")
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(lv1, x)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedCustom)


def test_leakyrelu():
    verify_unary("LeakyRelu", [32, 32])
    verify_unary("LeakyRelu", [32, 32], attrs={"alpha": 0.2})


def test_hardsigmoid():
    model = make_unary_model("HardSigmoid", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.20000000298023224, "float32"), x
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(lv, R.const(0.5, "float32"))
                gv: R.Tensor((2, 3), dtype="float32") = R.clip(
                    lv1, R.prim_value(0), R.prim_value(1)
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("HardSigmoid", [2, 3], attrs={"alpha": 0.3, "beta": 0.4})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.30000001192092896, "float32"), x
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(
                    lv, R.const(0.40000000596046448, "float32")
                )
                gv: R.Tensor((2, 3), dtype="float32") = R.clip(
                    lv1, R.prim_value(0), R.prim_value(1)
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedCustom)

    model = make_unary_model("HardSigmoid", [1, 3, 20, 20], attrs={"alpha": 0.5, "beta": 0.6})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom4D:
        @R.function
        def main(
            x: R.Tensor((1, 3, 20, 20), dtype="float32"),
        ) -> R.Tensor((1, 3, 20, 20), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 20, 20), dtype="float32") = R.multiply(
                    R.const(0.5, "float32"), x
                )
                lv1: R.Tensor((1, 3, 20, 20), dtype="float32") = R.add(
                    lv, R.const(0.60000002384185791, "float32")
                )
                gv: R.Tensor((1, 3, 20, 20), dtype="float32") = R.clip(
                    lv1, R.prim_value(0), R.prim_value(1)
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedCustom4D)


def test_shrink():
    model = make_unary_model("Shrink", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(x, R.const(0.5, "float32"))
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(x, R.const(0.0, "float32"))
                lv2: R.Tensor((2, 3), dtype="float32") = R.zeros_like(x, dtype="void")
                lv3: R.Tensor((2, 3), dtype="float32") = R.where(lv, lv1, lv2)
                lv4: R.Tensor((), dtype="float32") = R.negative(R.const(0.5, "float32"))
                lv5: R.Tensor((2, 3), dtype="bool") = R.less(x, lv4)
                lv6: R.Tensor((2, 3), dtype="float32") = R.add(x, R.const(0.0, "float32"))
                lv7: R.Tensor((2, 3), dtype="float32") = R.where(lv5, lv6, lv2)
                gv: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv7)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("Shrink", [2, 3], attrs={"lambd": 0.2, "bias": 0.1})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(
                    x, R.const(0.20000000298023224, "float32")
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(
                    x, R.const(0.10000000149011612, "float32")
                )
                lv2: R.Tensor((2, 3), dtype="float32") = R.zeros_like(x, dtype="void")
                lv3: R.Tensor((2, 3), dtype="float32") = R.where(lv, lv1, lv2)
                lv4: R.Tensor((), dtype="float32") = R.negative(
                    R.const(0.20000000298023224, "float32")
                )
                lv5: R.Tensor((2, 3), dtype="bool") = R.less(x, lv4)
                lv6: R.Tensor((2, 3), dtype="float32") = R.add(
                    x, R.const(0.10000000149011612, "float32")
                )
                lv7: R.Tensor((2, 3), dtype="float32") = R.where(lv5, lv6, lv2)
                gv: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv7)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedCustom)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("pad", [0, 2])
@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
def test_conv(stride: int, dilation: int, pad: int, bias: bool, auto_pad: str):
    def _verify_conv(input_shape, weight_shape):
        nd = len(weight_shape) - 2
        if auto_pad == "VALID":
            output_shape = [input_shape[0], weight_shape[0]] + [
                (input_shape[i] - dilation * (weight_shape[i] - 1) - 1) // stride + 1
                for i in range(2, len(input_shape))
            ]
            bias_shape = [output_shape[1]]
            conv_node = helper.make_node(
                "Conv",
                inputs=["x", "w"] + (["b"] if bias else []),
                outputs=["y"],
                strides=[stride] * nd,
                dilations=[dilation] * nd,
                auto_pad=auto_pad,
                group=input_shape[1] // weight_shape[1],
            )
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            if dilation == 2:
                # auto_pad = "SAME" and dilation = 2 is not supported in ONNX
                return
            output_shape = [input_shape[0], weight_shape[0]] + [
                (input_shape[i] + stride - 1) // stride for i in range(2, len(input_shape))
            ]
            bias_shape = [output_shape[1]]
            conv_node = helper.make_node(
                "Conv",
                inputs=["x", "w"] + (["b"] if bias else []),
                outputs=["y"],
                strides=[stride] * nd,
                dilations=[dilation] * nd,
                auto_pad=auto_pad,
                group=input_shape[1] // weight_shape[1],
            )
        else:
            output_shape = [input_shape[0], weight_shape[0]] + [
                (input_shape[i] + 2 * pad - dilation * (weight_shape[i] - 1) - 1) // stride + 1
                for i in range(2, len(input_shape))
            ]
            bias_shape = [output_shape[1]]
            conv_node = helper.make_node(
                "Conv",
                inputs=["x", "w"] + (["b"] if bias else []),
                outputs=["y"],
                strides=[stride] * nd,
                dilations=[dilation] * nd,
                pads=[pad] * nd * 2,
                group=input_shape[1] // weight_shape[1],
            )
        graph = helper.make_graph(
            [conv_node],
            "conv_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
            ]
            + ([helper.make_tensor_value_info("b", TensorProto.FLOAT, bias_shape)] if bias else []),
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="conv_test")
        check_correctness(model, atol=1e-4)

    # Conv1D
    _verify_conv([3, 4, 32], [4, 4, 3])
    _verify_conv([3, 4, 32], [2, 4, 3])  # group=2
    # Conv2D
    _verify_conv([3, 4, 32, 32], [4, 4, 3, 3])
    _verify_conv([3, 4, 32, 32], [2, 4, 3, 3])  # group=2
    # Conv3D
    _verify_conv([3, 4, 32, 32, 32], [4, 4, 3, 3, 3])
    _verify_conv([3, 4, 32, 32, 32], [2, 4, 3, 3, 3])  # group=2


@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("pad", [0, 2])
@pytest.mark.parametrize("output_pad", [0, 1])
def test_conv_transpose(stride: int, dilation: int, pad: int, bias: bool, output_pad: int):
    def _verify_conv_transpose(input_shape, weight_shape):
        nd = len(weight_shape) - 2
        output_shape = [input_shape[0], weight_shape[0]] + [
            (input_shape[i] - 1) * stride
            - 2 * pad
            + dilation * (weight_shape[i] - 1)
            + output_pad
            + 1
            for i in range(2, len(input_shape))
        ]
        bias_shape = [output_shape[1]]
        conv_node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "w"] + (["b"] if bias else []),
            outputs=["y"],
            strides=[stride] * nd,
            dilations=[dilation] * nd,
            pads=[pad] * nd * 2,
            output_padding=[output_pad] * nd,
            group=input_shape[1] // weight_shape[1],
        )
        graph = helper.make_graph(
            [conv_node],
            "conv_transpose_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
            ]
            + ([helper.make_tensor_value_info("b", TensorProto.FLOAT, bias_shape)] if bias else []),
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="conv_transpose_test")
        check_correctness(model, atol=1e-4)

    # ConvTranspose1D
    _verify_conv_transpose([3, 4, 32], [4, 4, 3])
    _verify_conv_transpose([3, 4, 32], [4, 2, 3])  # group=2
    # ConvTranspose2D
    _verify_conv_transpose([3, 4, 32, 32], [4, 4, 3, 3])
    _verify_conv_transpose([3, 4, 32, 32], [4, 2, 3, 3])  # group=2
    # ConvTranspose3D
    _verify_conv_transpose([3, 4, 12, 12, 12], [4, 4, 3, 3, 3])
    _verify_conv_transpose([3, 4, 12, 12, 12], [4, 2, 3, 3, 3])  # group=2


@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv_transpose_auto_pad(auto_pad: str, stride: int):
    def _verify(input_shape, weight_shape):
        nd = len(weight_shape) - 2
        conv_node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "w"],
            outputs=["y"],
            kernel_shape=weight_shape[2:],
            strides=[stride] * nd,
            auto_pad=auto_pad,
        )
        graph = helper.make_graph(
            [conv_node],
            "conv_transpose_auto_pad_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, producer_name="conv_transpose_auto_pad_test")
        check_correctness(model, atol=1e-4)

    # ConvTranspose1D / 2D / 3D
    _verify([1, 1, 8], [1, 1, 3])
    _verify([1, 1, 8, 8], [1, 1, 3, 3])
    _verify([1, 1, 4, 4, 4], [1, 1, 3, 3, 3])


def test_pow():
    verify_binary("Pow", [32, 32], [32, 32], [32, 32])


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("exclusive", [True, False])
def test_cumsum(reverse, exclusive):
    cumsum_node = helper.make_node(
        "CumSum", ["x", "axis"], ["y"], reverse=reverse, exclusive=exclusive
    )
    shape = [32, 32]
    graph = helper.make_graph(
        [cumsum_node],
        "cumsum_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=[helper.make_tensor("axis", TensorProto.INT64, (), [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(
        graph, producer_name="cumsum_test", opset_imports=[helper.make_opsetid("", 14)]
    )
    check_correctness(model)


def test_cumsum_int32_1d_axis_initializer():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
        ],
        initializer=[helper.make_tensor("axis", onnx.TensorProto.INT32, [1], [0])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_graph")
    check_correctness(model)


def test_cumsum_dynamic_axis_not_supported():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_dynamic_axis_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
            helper.make_tensor_value_info("axis", onnx.TensorProto.INT32, [1], "axis"),
        ],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_dynamic_axis_graph")
    with pytest.raises(ValueError, match="non-constant axis input is not supported"):
        from_onnx(model, opset=14, keep_params_in_input=True)


def test_cumsum_axis_shape_validation():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_invalid_axis_shape_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
        ],
        initializer=[helper.make_tensor("axis", onnx.TensorProto.INT64, [2], [0, 1])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_invalid_axis_shape_graph")
    with pytest.raises(
        ValueError,
        match=r"axis input must be a scalar \(0-D\) or a single-element 1-D tensor",
    ):
        from_onnx(model, opset=14, keep_params_in_input=True)


@pytest.mark.parametrize("axis", [[0, 2], None])
def test_squeeze(axis):
    if axis:
        squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    else:
        squeeze_node = helper.make_node("Squeeze", ["x"], ["y"])
    shape = [1, 32, 1, 32]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(
        graph, producer_name="squeeze_test", opset_imports=[helper.make_opsetid("", 13)]
    )
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    if axis:
        tvm_model["main"] = tvm_model["main"].without_attr("params")

    data = relax.Var("x", relax.TensorType(shape, "float32"))
    params = [data]
    if axis:
        params.append(relax.Var("axes", relax.TensorType([len(axis)], "int64")))

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            out = bb.emit_output(relax.op.squeeze(data, axis=axis))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("axis", [[0, 2], None])
def test_squeeze_constant(axis):
    shape = [1, 32, 1, 32]
    data = rg.standard_normal(size=shape).astype("float32")
    constant = make_constant_node("x", onnx.TensorProto.FLOAT, shape, data.flatten().tolist())
    if axis:
        squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    else:
        squeeze_node = helper.make_node("Squeeze", ["x"], ["y"])

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [constant, squeeze_node],
        "squeeze_test",
        inputs=[],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(
        graph, producer_name="squeeze_test", opset_imports=[helper.make_opsetid("", 13)]
    )
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    if axis:
        tvm_model["main"] = tvm_model["main"].without_attr("params")
    expected_data = np.squeeze(data, axis=tuple(axis) if axis else None)

    params = []
    if axis:
        params.append(relax.Var("axes", relax.TensorType([len(axis)], "int64")))
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            out = bb.emit_output(relax.const(expected_data, "float32"))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 0)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("axis", [[0]])
@pytest.mark.parametrize("A", [8, 16, 32])
@pytest.mark.parametrize("B", [8, 16, 32])
def test_dynamic_squeeze(axis, A, B):
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, "A", "B"]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["A", "B"])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    data_shape = _shape_with_size_vars(shape, "squeeze_input_dim")
    data = relax.Var("x", relax.TensorType(data_shape, "float32"))
    axes = relax.Var("axes", relax.TensorType([len(axis)], "int64"))
    bb = relax.BlockBuilder()
    with bb.function("main", [data, axes]):
        with bb.dataflow():
            out = bb.emit_output(relax.op.squeeze(data, axis=axis))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_squeeze_dynamic_axes_ir():
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_dynamic_axes_ir",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_dynamic_axes_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 32, 1, 32), dtype="float32"),
            axes: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=2):
            R.func_attr({"num_input": 2})
            squeeze_num_keep_dims = T.int64()
            squeeze_dim_0 = T.int64()
            squeeze_dim_1 = T.int64()
            with R.dataflow():
                lv: R.Shape([1, 32, 1, 32]) = R.shape_of(x)
                lv1: R.Tensor((2,), dtype="bool") = R.less(axes, R.const(0, "int64"))
                lv2: R.Tensor((2,), dtype="int64") = R.add(axes, R.const(4, "int64"))
                lv3: R.Tensor((4,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(4), R.prim_value(1), dtype="int64"
                )
                lv4: R.Tensor((2,), dtype="int64") = R.where(lv1, lv2, axes)
                lv5: R.Tensor((4, 1), dtype="int64") = R.expand_dims(lv3, axis=[1])
                lv6: R.Tensor((1, 2), dtype="int64") = R.expand_dims(lv4, axis=[0])
                lv7: R.Tensor((4, 2), dtype="bool") = R.equal(lv5, lv6)
                lv8: R.Tensor((4, 2), dtype="int64") = R.astype(lv7, dtype="int64")
                lv9: R.Tensor((4,), dtype="int64") = R.sum(lv8, axis=[1], keepdims=False)
                lv10: R.Tensor((4,), dtype="bool") = R.equal(lv9, R.const(0, "int64"))
                lv11: R.Tensor((1, squeeze_num_keep_dims), dtype="int64") = R.match_cast(
                    R.nonzero(lv10), R.Tensor((1, squeeze_num_keep_dims), dtype="int64")
                )
                lv12: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(lv)
                lv13: R.Tensor((squeeze_num_keep_dims,), dtype="int64") = R.reshape(
                    lv11, R.shape([squeeze_num_keep_dims])
                )
                lv14: R.Tensor((2,), dtype="int64") = R.match_cast(
                    R.take(lv12, lv13, axis=0, mode="fast"), R.Tensor((2,), dtype="int64")
                )
                lv15: R.Shape(ndim=2) = R.tensor_to_shape(lv14)
                lv16: R.Shape([squeeze_dim_0, squeeze_dim_1]) = R.match_cast(
                    lv15, R.Shape([squeeze_dim_0, squeeze_dim_1])
                )
                gv: R.Tensor((squeeze_dim_0, squeeze_dim_1), dtype="float32") = R.reshape(
                    x, R.shape([squeeze_dim_0, squeeze_dim_1])
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_squeeze_dynamic_axes_rank_validation():
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_dynamic_axes_rank_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1, 2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_dynamic_axes_rank_validation_test")
    with pytest.raises(ValueError, match="Expected a 1-D tensor"):
        from_onnx(model, opset=13, keep_params_in_input=True)


@pytest.mark.parametrize("axis", [[0]])
def test_dynamic_shape_squeeze(axis):
    shape_node = helper.make_node("Shape", ["x"], ["y"])
    squeeze_node = helper.make_node("Squeeze", ["y", "axes"], ["z"])
    shape = ["A"]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [shape_node, squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, [])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(("A",), dtype="float32"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> T.int64:
            A = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: T.int64 = R.prim_value(A)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_const():
    shape = [32, 32]
    const_value = np.random.rand(*shape).astype(np.float32)
    const_node = helper.make_node(
        "Constant",
        [],
        ["y"],
        value=helper.make_tensor("value", TensorProto.FLOAT, shape, const_value.flatten()),
    )
    graph = helper.make_graph(
        [const_node],
        "const_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="const_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((32, 32), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = R.const(const_value, "float32")
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_instance_norm():
    def verify_instance_norm(input_shape, scale_shape, bias_shape, expected):
        node = helper.make_node("InstanceNormalization", ["a", "b", "c"], ["d"], epsilon=1e-12)
        graph = helper.make_graph(
            [node],
            "instance_norm_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, scale_shape),
                helper.make_tensor_value_info("c", TensorProto.FLOAT, bias_shape),
            ],
            outputs=[helper.make_tensor_value_info("d", TensorProto.FLOAT, input_shape)],
        )

        model = helper.make_model(graph, producer_name="instance_norm_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class Expected4D:
        @R.function
        def main(
            a: R.Tensor((1, 3, 32, 32), dtype="float32"),
            b: R.Tensor((3,), dtype="float32"),
            c: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(a, axis=[2, 3], keepdims=True)
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = R.subtract(a, lv)
                lv2: R.Tensor((1, 3, 1, 1), dtype="float32") = R.variance(
                    a, axis=[2, 3], keepdims=True
                )
                lv3: R.Tensor((1, 3, 1, 1), dtype="float32") = R.add(lv2, R.const(1e-12, "float32"))
                lv4: R.Tensor((1, 3, 1, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((3, 1, 1), dtype="float32") = R.reshape(b, R.shape([3, 1, 1]))
                lv7: R.Tensor((1, 3, 32, 32), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((3, 1, 1), dtype="float32") = R.reshape(c, R.shape([3, 1, 1]))
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.add(lv7, lv8)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3D:
        @R.function
        def main(
            a: R.Tensor((1, 32, 32), dtype="float32"),
            b: R.Tensor((32,), dtype="float32"),
            c: R.Tensor((32,), dtype="float32"),
        ) -> R.Tensor((1, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 32, 1), dtype="float32") = R.mean(a, axis=[2], keepdims=True)
                lv1: R.Tensor((1, 32, 32), dtype="float32") = R.subtract(a, lv)
                lv2: R.Tensor((1, 32, 1), dtype="float32") = R.variance(a, axis=[2], keepdims=True)
                lv3: R.Tensor((1, 32, 1), dtype="float32") = R.add(lv2, R.const(1e-12, "float32"))
                lv4: R.Tensor((1, 32, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 32, 32), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((32, 1), dtype="float32") = R.reshape(b, R.shape([32, 1]))
                lv7: R.Tensor((1, 32, 32), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((32, 1), dtype="float32") = R.reshape(c, R.shape([32, 1]))
                gv: R.Tensor((1, 32, 32), dtype="float32") = R.add(lv7, lv8)
                R.output(gv)
            return gv

    verify_instance_norm([1, 3, 32, 32], [3], [3], Expected4D)
    verify_instance_norm([1, 32, 32], [32], [32], Expected3D)


def test_mean_variance_norm():
    def verify_mean_variance_norm(axes, expected):
        node = helper.make_node("MeanVarianceNormalization", ["x"], ["y"], axes=axes)
        graph = helper.make_graph(
            [node],
            "mean_variance_norm_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 32, 32])],
        )

        model = helper.make_model(graph, producer_name="mean_variance_norm_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedDefaultAxes:
        @R.function
        def main(
            x: R.Tensor((1, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(
                    x, axis=[0, 2, 3], keepdims=True
                )
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 3, 32, 32), dtype="float32") = R.power(x, R.const(2.0, "float32"))
                lv3: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(
                    lv2, axis=[0, 2, 3], keepdims=True
                )
                lv4: R.Tensor((1, 3, 1, 1), dtype="float32") = R.power(lv, R.const(2.0, "float32"))
                lv5: R.Tensor((1, 3, 1, 1), dtype="float32") = R.subtract(lv3, lv4)
                lv6: R.Tensor((1, 3, 1, 1), dtype="float32") = R.sqrt(lv5)
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(lv1, lv6)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedChannelAxes:
        @R.function
        def main(
            x: R.Tensor((1, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.mean(
                    x, axis=[1, 2, 3], keepdims=True
                )
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 3, 32, 32), dtype="float32") = R.power(x, R.const(2.0, "float32"))
                lv3: R.Tensor((1, 1, 1, 1), dtype="float32") = R.mean(
                    lv2, axis=[1, 2, 3], keepdims=True
                )
                lv4: R.Tensor((1, 1, 1, 1), dtype="float32") = R.power(lv, R.const(2.0, "float32"))
                lv5: R.Tensor((1, 1, 1, 1), dtype="float32") = R.subtract(lv3, lv4)
                lv6: R.Tensor((1, 1, 1, 1), dtype="float32") = R.sqrt(lv5)
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(lv1, lv6)
                R.output(gv)
            return gv

    verify_mean_variance_norm((0, 2, 3), ExpectedDefaultAxes)
    verify_mean_variance_norm((1, 2, 3), ExpectedChannelAxes)


def test_layer_norm():
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)

    # Test case with no bias that is an optional input
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)

    # No bias with a non-square input where data.shape[1] differs from the scale
    # shape, see https://github.com/apache/tvm/issues/19691.
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)

    # No bias with a non-square fp16 input. The synthesized zero bias must match
    # the scale dtype, otherwise layer_norm rejects the float32 bias, see
    # https://github.com/apache/tvm/issues/19691.
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model, opset=17, atol=1e-2, rtol=1e-2)

    # Same no-bias path for bf16. ONNX Runtime's CPU provider has no bf16
    # LayerNormalization kernel, so this only checks the importer builds the
    # graph with a bf16 zero bias (the dtype the fix derives from the scale).
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.BFLOAT16, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    model.opset_import[0].version = 17
    from_onnx(model, opset=17, keep_params_in_input=True)


def test_layer_norm_with_nd_gamma_beta():
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], axis=1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_with_nd_gamma_beta_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3, 4, 4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3, 4, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_with_nd_gamma_beta_test")
    check_correctness(model)

    # Test case with no bias that is an optional input
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_with_nd_gamma_beta_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_with_nd_gamma_beta_test")
    check_correctness(model)


def test_layer_norm_numerical_stability():
    """Numerical stability test for https://github.com/apache/tvm/issues/19592."""
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], axis=-1, epsilon=1e-5
    )
    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_numerical_stability",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4]),
        ],
    )
    model = helper.make_model(graph, producer_name="layer_norm_numerical_stability")

    input_array = np.array([[80000.0, 80001.0, 80002.0, 80003.0]], dtype=np.float32)
    scale_array = np.ones(4, dtype=np.float32)
    bias_array = np.zeros(4, dtype=np.float32)
    inputs = {"input": input_array, "scale": scale_array, "bias": bias_array}

    # ONNXRuntime also returns NaN for Large-value, small-variance inputs, so we here
    # compare against a two-pass reference instead of ORT.
    mean = input_array.mean(axis=-1, keepdims=True)
    var = ((input_array - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = ((input_array - mean) / np.sqrt(var + 1e-5) * scale_array + bias_array).astype(
        np.float32
    )

    tvm_output = run_in_tvm(model, inputs=inputs, ir_version=9, opset=17)

    assert np.isfinite(tvm_output.numpy()).all()
    tvm.testing.assert_allclose(tvm_output.numpy(), expected)


def test_rms_norm():
    # Basic test: default axis=-1
    rms_norm_node = helper.make_node("RMSNormalization", ["input", "scale"], ["Y"], epsilon=1e-05)

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 8, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_test")
    check_correctness(model, opset=23)

    # Test with explicit axis=1 (normalize over last 2 dims)
    rms_norm_node = helper.make_node(
        "RMSNormalization", ["input", "scale"], ["Y"], axis=1, epsilon=1e-06
    )

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_axis_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8, 16]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [8, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8, 16]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_axis_test")
    check_correctness(model, opset=23)

    # Test with float16 input (stash_type=1 means compute in float32)
    rms_norm_node = helper.make_node(
        "RMSNormalization", ["input", "scale"], ["Y"], epsilon=1e-05, stash_type=1
    )

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_fp16_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 8, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 8, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_fp16_test")
    check_correctness(model, opset=23, rtol=1e-2, atol=1e-2)


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [False])
def test_skiplayernormalization(dynamic):
    def verify_skiplayernormalization(input_, skip, gamma, beta, bias):
        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["input", "skip", "gamma", "beta", "bias"],
            outputs=["output", "mean", "std_dev"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        input_shape = list(input_.shape)
        skip_shape = list(skip.shape)
        gamma_shape = list(gamma.shape)
        beta_shape = list(beta.shape)
        bias_shape = list(bias.shape)
        output_shape = list(input_.shape)
        mean_shape = list([1])
        std_dev_shape = list([1])
        if dynamic:
            input_shape = ["?" for _ in range(len(input_.shape))]
            skip_shape = ["?" for _ in range(len(skip.shape))]
            gamma_shape = ["?" for _ in range(len(gamma.shape))]
            beta_shape = ["?" for _ in range(len(beta.shape))]
            bias_shape = ["?" for _ in range(len(bias.shape))]
            output_shape = ["?" for _ in range(len(input_.shape))]

        graph = helper.make_graph(
            [node],
            "skiplayernormalization_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, skip_shape),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, gamma_shape),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, beta_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean_shape),
                helper.make_tensor_value_info("std_dev", TensorProto.FLOAT, std_dev_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="skiplayernormalization_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)

        input_var = relax.Var("input", relax.TensorType(input_shape, "float32"))
        skip_var = relax.Var("skip", relax.TensorType(skip_shape, "float32"))
        gamma_var = relax.Var("gamma", relax.TensorType(gamma_shape, "float32"))
        beta_var = relax.Var("beta", relax.TensorType(beta_shape, "float32"))
        bias_var = relax.Var("bias", relax.TensorType(bias_shape, "float32"))

        bb = relax.BlockBuilder()
        with bb.function("main", [input_var, skip_var, gamma_var, beta_var, bias_var]):
            with bb.dataflow():
                skipped = bb.emit(relax.op.add(input_var, skip_var))
                biased = bb.emit(relax.op.add(skipped, bias_var))
                output = bb.emit(
                    relax.op.nn.layer_norm(
                        biased,
                        gamma_var,
                        beta_var,
                        axes=-1,
                        epsilon=float(np.float32(1e-4)),
                    )
                )
                gv = bb.emit_output(
                    relax.Tuple([output, relax.const(0, "float32"), relax.const(0, "float32")])
                )
            bb.emit_func_output(gv)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 5)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    hidden_size = 384
    batch_size = 4
    sequence_length = 4

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    skip = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype(dtype)
    beta = np.random.randn(hidden_size).astype(dtype) * 0.1
    bias = np.random.randn(hidden_size).astype(dtype)

    verify_skiplayernormalization(input_array, skip, gamma, beta, bias)


def test_embedlayernormalization():
    def verify_embedlayernormalization(
        input_ids,
        segment_ids,
        word_embedding,
        position_embedding,
        segment_embedding,
        gamma,
        beta,
    ):
        node = onnx.helper.make_node(
            "EmbedLayerNormalization",
            inputs=[
                "input_ids",
                "" if segment_ids is None else "segment_ids",
                "word_embedding",
                "position_embedding",
                "" if segment_embedding is None else "segment_embedding",
                "gamma",
                "beta",
            ],
            outputs=["output", "mask_index"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        segment_ids_shape = [] if segment_ids is None else segment_ids.shape
        segment_embedding_shape = [] if segment_embedding is None else segment_embedding.shape

        graph = helper.make_graph(
            [node],
            "embedlayernormalization_test",
            inputs=[
                helper.make_tensor_value_info(
                    "input_ids", TensorProto.INT32, list(input_ids.shape)
                ),
                helper.make_tensor_value_info("segment_ids", TensorProto.INT32, segment_ids_shape),
                helper.make_tensor_value_info(
                    "word_embedding", TensorProto.FLOAT, list(word_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "position_embedding", TensorProto.FLOAT, list(position_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "segment_embedding", TensorProto.FLOAT, segment_embedding_shape
                ),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, list((batch_size, sequence_length, hidden_size))
                ),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch_size]),
            ],
        )

        model = helper.make_model(graph, producer_name="embedlayernormalization_test")

        tvm_model = from_onnx(model, keep_params_in_input=True)

        input_ids_var = relax.Var("input_ids", relax.TensorType(list(input_ids.shape), "int32"))
        segment_ids_var = relax.Var("segment_ids", relax.TensorType(segment_ids_shape, "int32"))
        word_embedding_var = relax.Var(
            "word_embedding", relax.TensorType(list(word_embedding.shape), "float32")
        )
        position_embedding_var = relax.Var(
            "position_embedding", relax.TensorType(list(position_embedding.shape), "float32")
        )
        segment_embedding_var = relax.Var(
            "segment_embedding", relax.TensorType(segment_embedding_shape, "float32")
        )
        gamma_var = relax.Var("gamma", relax.TensorType(list(gamma.shape), "float32"))
        beta_var = relax.Var("beta", relax.TensorType(list(beta.shape), "float32"))

        bb = relax.BlockBuilder()
        with bb.function(
            "main",
            [
                input_ids_var,
                segment_ids_var,
                word_embedding_var,
                position_embedding_var,
                segment_embedding_var,
                gamma_var,
                beta_var,
            ],
        ):
            with bb.dataflow():
                word_vec = bb.emit(relax.op.take(word_embedding_var, input_ids_var, axis=0))
                pos_ids = relax.const([list(range(sequence_length))] * batch_size, dtype="int64")
                pos_vec = bb.emit(relax.op.take(position_embedding_var, pos_ids, axis=0))
                vec_sum = bb.emit(relax.op.add(word_vec, pos_vec))
                if segment_ids is not None:
                    segment_vec = bb.emit(
                        relax.op.take(segment_embedding_var, segment_ids_var, axis=0)
                    )
                    vec_sum = bb.emit(relax.op.add(vec_sum, segment_vec))
                output = bb.emit(
                    relax.op.nn.layer_norm(
                        vec_sum,
                        gamma_var,
                        beta_var,
                        axes=-1,
                        epsilon=float(np.float32(1e-4)),
                    )
                )
                gv = bb.emit_output(
                    relax.Tuple(
                        [
                            output,
                            relax.const(np.zeros((batch_size,), dtype="int64")),
                        ]
                    )
                )
            bb.emit_func_output(gv)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 7)

        tvm.ir.assert_structural_equal(tvm_model, expected)

        # TODO(@anwang2009): onnxruntime v1.9.0 requires empty list for optional argument,
        # but v1.10.0+ requires None instead.
        # verify_with_ort_with_inputs(
        #     model,
        #     [
        #         input_ids,
        #         np.empty(0, dtype="int32") if segment_ids is None else segment_ids,
        #         word_embedding,
        #         position_embedding,
        #         np.empty(0, dtype="float32") if segment_embedding is None else segment_embedding,
        #         gamma,
        #         beta,
        #     ],
        #     [
        #         (batch_size, sequence_length, hidden_size),
        #         batch_size,
        #     ],
        #     target=target,
        #     dev=dev,
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

    hidden_size = 384
    batch_size = 4
    sequence_length = 3
    vocab_size = 5

    input_ids = np.full((batch_size, sequence_length), 3).astype("int32")
    segment_ids = np.zeros((batch_size, sequence_length)).astype("int32")
    word_embedding = np.full((vocab_size, hidden_size), 1).astype("float32")
    position_embedding = np.full((sequence_length, hidden_size), 2).astype("float32")
    segment_embedding = np.full((vocab_size, hidden_size), 3).astype("float32")

    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype("float32")
    beta = np.random.randn(hidden_size).astype("float32") * 0.1

    verify_embedlayernormalization(
        input_ids, segment_ids, word_embedding, position_embedding, segment_embedding, gamma, beta
    )

    # Test with undefined segment embedding
    verify_embedlayernormalization(
        input_ids, None, word_embedding, position_embedding, None, gamma, beta
    )


def test_local_response_norm():
    lrn_node = helper.make_node(
        op_type="LRN",
        inputs=["input"],
        outputs=["output"],
        name="LRN_Node",
        alpha=0.0001,
        beta=0.75,
        bias=1.0,
        size=3,
    )

    graph = helper.make_graph(
        [lrn_node],
        "local_response_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="local_response_norm_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((1, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.multiply(input, input)
                lv1: R.Tensor((1, 1, 3, 32, 32), dtype="float32") = R.expand_dims(lv, axis=[1])
                lv2: R.Tensor((1, 1, 3, 32, 32), dtype="float32") = R.nn.avg_pool3d(
                    lv1,
                    pool_size=[3, 1, 1],
                    strides=[1, 1, 1],
                    dilation=[1, 1, 1],
                    padding=[1, 0, 0, 1, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                lv3: R.Tensor((1, 3, 32, 32), dtype="float32") = R.squeeze(lv2, axis=[1])
                lv4: R.Tensor((1, 3, 32, 32), dtype="float32") = R.multiply(
                    lv3, R.const(9.9999997473787516e-05, "float32")
                )
                lv5: R.Tensor((1, 3, 32, 32), dtype="float32") = R.add(lv4, R.const(1.0, "float32"))
                lv6: R.Tensor((1, 3, 32, 32), dtype="float32") = R.power(
                    lv5, R.const(0.75, "float32")
                )
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(input, lv6)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


COMPOSITE_REDUCE_FUNCS = [
    "ReduceSumSquare",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceL1",
    "ReduceL2",
]

REDUCE_AXES_ATTR_TEST_CASES = [
    ([3, 2, 2], None),
    ([3, 2, 3], None),
    ([3, 3, 3], (1,)),
    ([3, 3, 3, 1], (1, 2)),
    ([3, 3, 3, 1], (1,)),
    ([1, 3, 4, 1], (1,)),
]

REDUCE_AXES_INPUT_TEST_CASES = [
    ([3, 2, 2], [], False),
    ([3, 2, 2], None, False),
    ([4, 3], [], True),
    ([3, 3, 3, 1], (1, 2), False),
]


def _reduce_output_shape(input_shape: list[int], axes, keepdims: bool, noop_with_empty_axes=False):
    if noop_with_empty_axes and not axes:
        return list(input_shape)
    axis = None if not axes else axes
    return list(np.sum(np.empty(input_shape), axis=axis, keepdims=keepdims).shape)


def _make_composite_reduce_expected(
    func: str,
    input_shape: list[int],
    axes,
    keepdims: bool,
    dynamic: bool,
    axes_input_shape: list[int] | None = None,
    noop_with_empty_axes: bool = False,
):
    if dynamic:
        expected_input_shape = [
            tvm.tirx.SizeVar(f"reduce_dim_{i}", "int64") for i in range(len(input_shape))
        ]
    else:
        expected_input_shape = input_shape

    x = relax.Var("x", relax.TensorType(expected_input_shape, "float32"))
    params = [x]
    if axes_input_shape is not None:
        params.append(relax.Var("reduce_axes", relax.TensorType(axes_input_shape, "int64")))

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            if noop_with_empty_axes and not axes:
                out = bb.emit_output(x)
            elif func == "ReduceLogSumExp":
                max_x = bb.emit(relax.op.max(x, axes, True))
                exp_x = bb.emit(relax.op.exp(relax.op.subtract(x, max_x)))
                sum_x = bb.emit(relax.op.sum(exp_x, axes, True))
                if keepdims:
                    out = bb.emit_output(relax.op.add(relax.op.log(sum_x), max_x))
                else:
                    out_x = bb.emit(relax.op.add(relax.op.log(sum_x), max_x))
                    out = bb.emit_output(relax.op.squeeze(out_x, axes))
            elif func == "ReduceLogSum":
                sum_x = bb.emit(relax.op.sum(x, axes, keepdims))
                out = bb.emit_output(relax.op.log(sum_x))
            elif func == "ReduceSumSquare":
                square_x = bb.emit(relax.op.multiply(x, x))
                out = bb.emit_output(relax.op.sum(square_x, axes, keepdims))
            elif func == "ReduceL1":
                abs_x = bb.emit(relax.op.abs(x))
                out = bb.emit_output(relax.op.sum(abs_x, axes, keepdims))
            else:
                square_x = bb.emit(relax.op.multiply(x, x))
                sum_x = bb.emit(relax.op.sum(square_x, axes, keepdims))
                out = bb.emit_output(relax.op.sqrt(sum_x))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)
    return expected


def create_reduce_test_parameters_axes_attr():
    output = []
    for value in [True, False]:
        output.append(("ReduceMax", value, 11))
        output.append(("ReduceMean", value, 13))
        output.append(("ReduceMin", value, 11))
        output.append(("ReduceProd", value, 13))
        output.append(("ReduceSum", value, 11))
        # Opset 11-12 axes-as-attr: verifies get_converter does not
        # underflow to the v18 (axes-as-input) implementation.
        output.append(("ReduceMean", value, 11))
        output.append(("ReduceProd", value, 11))
    return output


def create_composite_reduce_test_parameters_axes_attr():
    output = []
    for dynamic in [True, False]:
        for opset in [13, 11]:
            for func in COMPOSITE_REDUCE_FUNCS:
                output.append((func, dynamic, opset))
    return output


@pytest.mark.parametrize("func, dynamic, opset", create_reduce_test_parameters_axes_attr())
def test_all_reduce_funcs_axes_attr(func, dynamic, opset):
    def verify_reduce_func(func, data, axis, keepdims):
        inshape = data.shape
        outshape = np.sum(data, axis=axis, keepdims=keepdims == 1).shape

        if axis:
            node = onnx.helper.make_node(
                func, inputs=["x"], outputs=["y"], axes=axis, keepdims=keepdims
            )
        else:
            node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], keepdims=keepdims)

        if dynamic:
            in_list = ["?" for _ in range(len(inshape))]
            out_list = ["?" for _ in range(len(outshape))]
        else:
            in_list = list(inshape)
            out_list = list(outshape)
        graph = helper.make_graph(
            [node],
            "reduce_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_list)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_list)],
        )

        model = helper.make_model(graph, producer_name="reduce_test")
        inputs_dict = {"x": data}
        # Reduction ops accumulate arithmetic errors, so we use a higher tolerance.
        check_correctness(model, inputs_dict, opset=opset, rtol=1e-4, atol=1e-4)

    for keepdims in [True, False]:
        verify_reduce_func(
            func, np.random.randn(3, 2, 2).astype(np.float32), axis=None, keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 2, 3).astype(np.float32), axis=None, keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3).astype(np.float32), axis=(1,), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1, 2), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1,), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(1, 3, 4, 1).astype(np.float32), axis=(1,), keepdims=keepdims
        )


@pytest.mark.parametrize(
    "func, dynamic, opset", create_composite_reduce_test_parameters_axes_attr()
)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("input_shape, axes", REDUCE_AXES_ATTR_TEST_CASES)
def test_composite_reduce_funcs_axes_attr_ir(func, dynamic, opset, keepdims, input_shape, axes):
    attrs = {"keepdims": keepdims}
    if axes:
        attrs["axes"] = axes
    node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], **attrs)
    output_shape = _reduce_output_shape(input_shape, axes, keepdims)
    graph = helper.make_graph(
        [node],
        "composite_reduce_axes_attr_ir_test",
        inputs=[
            helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, ["?"] * len(input_shape) if dynamic else input_shape
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, ["?"] * len(output_shape) if dynamic else output_shape
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="composite_reduce_axes_attr_ir_test",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

    tvm.ir.assert_structural_equal(
        tvm_model,
        _make_composite_reduce_expected(func, input_shape, axes, keepdims, dynamic),
    )


def create_reduce_test_parameters_axes_input():
    output = []
    for dynamic in [True, False]:
        output.append(("ReduceMax", dynamic, 18))
        output.append(("ReduceMean", dynamic, 18))
        output.append(("ReduceMin", dynamic, 18))
        output.append(("ReduceProd", dynamic, 18))
        output.append(("ReduceSum", dynamic, 13))
    return output


def create_composite_reduce_test_parameters_axes_input():
    output = []
    for dynamic in [True, False]:
        for func in COMPOSITE_REDUCE_FUNCS:
            output.append((func, dynamic, 18))
    return output


@pytest.mark.parametrize("func, dynamic, opset", create_reduce_test_parameters_axes_input())
def test_all_reduce_funcs_axes_input(func, dynamic, opset):
    def verify_reduce_func(func, data, axes, keepdims, noop_with_empty_axes=False):
        inshape = data.shape
        inputs = ["x"]
        initializers = []

        # Optional `axes` input
        if axes is not None:
            axes_name = "reduce_axes"
            axes_np = np.asarray(axes, dtype=np.int64)
            axes_init = helper.make_tensor(
                name=axes_name,
                data_type=TensorProto.INT64,
                dims=axes_np.shape,
                vals=axes_np,
            )
            initializers.append(axes_init)
            inputs.append(axes_name)

        # Determine input and output shapes
        if not axes and not noop_with_empty_axes:
            outshape = np.sum(data, axis=None, keepdims=keepdims).shape
        elif not axes and noop_with_empty_axes:
            outshape = inshape
        else:
            outshape = np.sum(data, axis=axes, keepdims=keepdims).shape

        if dynamic:
            in_list = ["?"] * len(inshape)
            out_list = ["?"] * len(outshape)
        else:
            in_list = list(inshape)
            out_list = list(outshape)

        # Make a model node
        node = helper.make_node(
            func,
            inputs=inputs,
            outputs=["y"],
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

        # Make a model graph and a model
        graph = helper.make_graph(
            [node],
            "reduce18_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_list)],
            initializer=initializers,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_list)],
        )
        model = helper.make_model(graph, producer_name="reduce18_test")

        inputs_dict = {"x": data}
        check_correctness(model, inputs_dict, opset=opset, rtol=1e-4, atol=1e-4)

    # Verify
    for keepdims in [True, False]:
        # no `axes` input && `noop_with_empty_axes` = 0 -> reduce over all dimensions.
        verify_reduce_func(
            func,
            np.random.randn(3, 2, 2).astype(np.float32),
            axes=[],
            keepdims=keepdims,
            noop_with_empty_axes=False,
        )

        # no `axes` input && `noop_with_empty_axes` = 0 -> reduce over all dimensions.
        verify_reduce_func(
            func,
            np.random.randn(3, 2, 2).astype(np.float32),
            axes=None,
            keepdims=keepdims,
            noop_with_empty_axes=False,
        )

        # no `axes` input && `noop_with_empty_axes` = 1 -> return the input unchanged.
        verify_reduce_func(
            func,
            np.random.randn(4, 3).astype(np.float32),
            axes=[],
            keepdims=keepdims,
            noop_with_empty_axes=True,
        )

        # no `axes` input && `noop_with_empty_axes` = 1 -> return the input unchanged.
        # (onnxruntime bug) Runtime error on the onnxruntime part
        # verify_reduce_func(
        #     func,
        #     np.random.randn(4, 3).astype(np.float32),
        #     axes=None,
        #     keepdims=keepdims,
        #     noop_with_empty_axes=True,
        # )

        # `axes` provided -> reduce over specified axes.
        verify_reduce_func(
            func,
            np.random.randn(3, 3, 3, 1).astype(np.float32),
            axes=(1, 2),
            keepdims=keepdims,
        )


@pytest.mark.parametrize(
    "func, dynamic, opset", create_composite_reduce_test_parameters_axes_input()
)
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("input_shape, axes, noop_with_empty_axes", REDUCE_AXES_INPUT_TEST_CASES)
def test_composite_reduce_funcs_axes_input_ir(
    func, dynamic, opset, keepdims, input_shape, axes, noop_with_empty_axes
):
    node_inputs = ["x"]
    initializers = []
    axes_input_shape = None
    if axes is not None:
        axes_np = np.asarray(axes, dtype=np.int64)
        axes_input_shape = list(axes_np.shape)
        initializers.append(
            helper.make_tensor(
                name="reduce_axes",
                data_type=TensorProto.INT64,
                dims=axes_input_shape,
                vals=axes_np,
            )
        )
        node_inputs.append("reduce_axes")

    effective_axes = None if not axes and not noop_with_empty_axes else axes
    output_shape = _reduce_output_shape(
        input_shape, effective_axes, keepdims, noop_with_empty_axes=noop_with_empty_axes
    )
    node = onnx.helper.make_node(
        func,
        inputs=node_inputs,
        outputs=["y"],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )
    graph = helper.make_graph(
        [node],
        "composite_reduce_axes_input_ir_test",
        inputs=[
            helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, ["?"] * len(input_shape) if dynamic else input_shape
            )
        ],
        initializer=initializers,
        outputs=[
            helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, ["?"] * len(output_shape) if dynamic else output_shape
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="composite_reduce_axes_input_ir_test",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    if axes_input_shape is not None:
        assert len(tvm_model["main"].attrs["params"]) == 1
        tvm_model["main"] = tvm_model["main"].without_attr("params")

    tvm.ir.assert_structural_equal(
        tvm_model,
        _make_composite_reduce_expected(
            func,
            input_shape,
            effective_axes,
            keepdims,
            dynamic,
            axes_input_shape=axes_input_shape,
            noop_with_empty_axes=noop_with_empty_axes,
        ),
    )


@pytest.mark.parametrize("in_dtype", [np.float32, np.int32])
@pytest.mark.parametrize("axis", [None, 0, 1, 2])
@pytest.mark.parametrize("keepdims", [None, True, False])
def test_arg_min_max(in_dtype, axis, keepdims):
    def verify_arg_min_max(input_dim, in_dtype, op_name="ArgMax", axis=None, keepdims=None):
        a_np1 = np.random.uniform(-10, 10, input_dim).astype(in_dtype)
        out_shape = list(a_np1.shape)
        def_axis = axis if axis is not None else 0
        if keepdims == 1 or keepdims is None:
            out_shape[def_axis] = 1
        else:
            out_shape.pop(def_axis)

        node = helper.make_node(op_name, inputs=["a_np1"], outputs=["out"])

        if keepdims is not None:
            keepdims_attr = helper.make_attribute("keepdims", keepdims)
            node.attribute.append(keepdims_attr)
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            node.attribute.append(axis_attr)

        graph = helper.make_graph(
            [node],
            "argreduce_test",
            inputs=[helper.make_tensor_value_info("a_np1", TensorProto.INT32, list(a_np1.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="arg_min_max_test")
        check_correctness(model)

    verify_arg_min_max([3, 4, 4], in_dtype, "ArgMax", axis, keepdims)
    verify_arg_min_max([3, 4, 4], in_dtype, "ArgMin", axis, keepdims)


@pytest.mark.parametrize("axis", [-1, 0, 1])
@pytest.mark.parametrize("largest", [True, False])
def test_topk(axis: int, largest: int):
    in_shape = [32, 32, 32]
    k_value = 4
    out_shape = in_shape
    out_shape[axis] = k_value
    k = make_constant_node("k", TensorProto.INT64, [1], [k_value])
    node = onnx.helper.make_node(
        "TopK",
        inputs=["data", "k"],
        outputs=["values", "indices"],
        axis=axis,
        largest=largest,
    )
    graph = helper.make_graph(
        [k, node],
        "topk_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape)],
        outputs=[
            helper.make_tensor_value_info("values", TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, out_shape),
        ],
    )
    model = helper.make_model(graph, producer_name="topk_test")

    check_correctness(model)


@pytest.mark.parametrize("dynamic", [False, True])
def test_expand(dynamic):
    def _assert_expand_ir(name, input_shape, target_shape, output_shape):
        shape_array = np.array(target_shape)
        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten().astype("int64"),
            ),
        )
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_teint64st",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        tvm_model = from_onnx(model, keep_params_in_input=True)

        x = relax.Var("in", relax.TensorType(input_shape, "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x]):
            with bb.dataflow():
                out = bb.emit_output(relax.op.broadcast_to(x, relax.ShapeExpr(output_shape)))
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 1)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    def _assert_expand_dynamic_shapeexpr_ir(name, input_shape, shape_input_shape):
        shape_node = onnx.helper.make_node("Shape", inputs=["in_2"], outputs=["shape"])
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])
        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_test",
            inputs=[
                helper.make_tensor_value_info("in", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("in_2", TensorProto.FLOAT, shape_input_shape),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, shape_input_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        tvm_model = from_onnx(model, keep_params_in_input=True)

        shape = [
            tvm.tirx.SizeVar(dim, "int64") if isinstance(dim, str) else dim
            for dim in shape_input_shape
        ]
        x = relax.Var("in", relax.TensorType(input_shape, "float32"))
        shape_source = relax.Var("in_2", relax.TensorType(shape, "float32"))
        bb = relax.BlockBuilder()
        with bb.function("main", [x, shape_source]):
            with bb.dataflow():
                out = bb.emit_output(relax.op.broadcast_to(x, relax.ShapeExpr(shape)))
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 2)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    if not dynamic:
        _assert_expand_ir("expand_with_dim_unchanged_test", [3, 1], [3, 4], [3, 4])
        _assert_expand_ir("expand_with_diff_dim", [3, 1], [1, 3, 4], [1, 3, 4])
        _assert_expand_ir("expand_with_the_same_suffix_dims", [3, 1], [1, 1, 3, 1], [1, 1, 3, 1])
    else:
        _assert_expand_dynamic_shapeexpr_ir(
            "expand_with_dynamic_dim", [1, 32, 32], ["batch", 32, 32]
        )


def test_expand_incompatible_broadcasting():
    """
    This test case reproduces the error where input tensor shape at dim 1 is 25
    and target shape at dim 3 is 56, which violates ONNX broadcasting rules
    """

    def _test_expand_error_case(name, data_shape, target_shape_vals):
        data = np.random.uniform(size=data_shape).astype(np.float32)

        shape_array = np.array(target_shape_vals, dtype=np.int64)
        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten(),
            ),
        )

        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_error_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, target_shape_vals)],
        )

        model = helper.make_model(graph, producer_name=name)

        with pytest.raises(ValueError) as exc_info:
            from_onnx(model, keep_params_in_input=True)

        error_msg = str(exc_info.value)
        assert "broadcast" in error_msg.lower() or "incompatible" in error_msg.lower(), (
            f"Expected broadcasting error, but got: {error_msg}"
        )

    # Test case 1: Reproduce the exact error from the issue-17769
    # Input shape: (25,), target shape: (1, 1, 1, 56)
    # This should faill because input dim 1 (25) != target dim 3 (56) and neither is 1
    _test_expand_error_case(
        "expand_incompatible_25_to_56",
        data_shape=(25,),
        target_shape_vals=(1, 1, 1, 56),
    )

    # Test case 2: Another incompatible case
    # Input shape: (1, 25), target shape: (1, 1, 1, 56)
    # After right-alignment, input (1, 1, 1, 25) vs. target (1, 1, 1, 56)
    # This should fail because 25 != 56 and neither is 1
    _test_expand_error_case(
        "expand_incompatible_aligned_25_to_56",
        data_shape=(1, 25),
        target_shape_vals=(1, 1, 1, 56),
    )

    # Test case 3: Valid case for comparison - should not raise error
    def _test_expand_valid_case():
        """Test a valid expand case to ensure our fix doesn't break valid operations"""
        data_shape = (1, 25)
        target_shape_vals = [2, 25]  # Valid: input (1, 25) can broadcast to (2, 25)

        data = np.random.uniform(size=data_shape).astype(np.float32)
        shape_array = np.array(target_shape_vals, dtype=np.int64)

        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten(),
            ),
        )

        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_valid_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, target_shape_vals)],
        )

        model = helper.make_model(graph, producer_name="expand_valid_test_case")

        try:
            tvm_model = from_onnx(model, keep_params_in_input=True)
        except Exception as e:
            pytest.fail(f"Valid expand case should not fail, but got error: {e}")

    _test_expand_valid_case()


# TODO(jwfromm) Current approach to dynamic expand is technically not well formed. Reenable once fixed.
@pytest.mark.skip("Produces ill-formed IR")
def test_constantofshape():
    def verify_constantofshape(input_dim, value, dtype):
        fill_node = helper.make_node(
            "ConstantOfShape",
            ["input"],
            ["output"],
            value=helper.make_tensor(
                "value", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), (1,), (value,)
            ),
        )

        inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [len(input_dim)])]

        graph = helper.make_graph(
            [fill_node],
            "fill_test",
            inputs,
            initializer=[
                helper.make_tensor(
                    "input",
                    TensorProto.INT64,
                    [len(input_dim)],
                    np.asarray(input_dim).astype("int64"),
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), input_dim
                )
            ],
        )

        model = helper.make_model(graph, producer_name="fill_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        assert tuple(dim.value for dim in tvm_model["main"].ret_ty.shape.values) == input_dim

    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


def test_constantofshape_default_value():
    """ConstantOfShape value attribute should default to float32 zero."""
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node("ConstantOfShape", ["shape"], ["y"])
    graph = helper.make_graph(
        [node],
        "constantofshape_default_value_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, producer_name="constantofshape_default_value_test")

    tvm_model = from_onnx(model)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.broadcast_to(
                    R.const(0.0, "float32"), R.shape([2, 3])
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_slice():
    def verify_slice(data_shape, output_shape, starts, ends, axes=None, steps=None):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")
        if isinstance(steps, list):
            steps = np.array(steps, "int64")

        slice_inputs = ["x", "starts", "ends"]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
        ]

        if axes is not None:
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes))
            slice_inputs.append("axes")
        if steps is not None:
            initializer.append(helper.make_tensor("steps", TensorProto.INT64, steps.shape, steps))
            slice_inputs.append("steps")

        slice_node = helper.make_node("Slice", inputs=slice_inputs, outputs=["y"])

        graph = helper.make_graph(
            [slice_node],
            "slice_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializer=initializer,
        )

        model = helper.make_model(graph, producer_name="slice_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_model["main"] = tvm_model["main"].without_attr("params")

        data = relax.Var("x", relax.TensorType(data_shape, "float32"))
        params = [
            data,
            relax.Var("starts", relax.TensorType(list(starts.shape), "int64")),
            relax.Var("ends", relax.TensorType(list(ends.shape), "int64")),
        ]

        axes_list = axes.tolist() if axes is not None else list(range(len(starts)))
        steps_list = steps.tolist() if steps is not None else [1] * len(axes_list)
        if axes is not None:
            params.append(relax.Var("axes", relax.TensorType(list(axes.shape), "int64")))
        if steps is not None:
            params.append(relax.Var("steps", relax.TensorType(list(steps.shape), "int64")))

        bb = relax.BlockBuilder()
        with bb.function("main", params):
            with bb.dataflow():
                out = bb.emit_output(
                    relax.op.strided_slice(
                        data,
                        axes_list,
                        starts.tolist(),
                        ends.tolist(),
                        steps_list,
                        assume_inbound=False,
                    )
                )
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 1)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    # Test with all parameters set.
    verify_slice([20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10], axes=[0, 1], steps=[1, 1])
    # Test with default axes and steps.
    verify_slice([20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10])
    # Test with negative steps.
    verify_slice(
        [20, 10, 5],
        [19, 3, 2],
        starts=[20, 10, 4],  # NOTE: the start is out of bounds
        ends=[0, 0, 1],
        steps=[-1, -3, -2],
        axes=[0, 1, 2],
    )
    verify_slice([20, 10, 5], [10, 5], starts=[0, 0], ends=[3, 10], axes=[1, 2])
    verify_slice([20, 10, 5], [10, 5], starts=[0, 0], ends=[3, 10], axes=[1, 2])

    # TODO (gigiblender): Enable this test when we have a way to pass the steps but not axes.
    # verify_slice(
    #     [20, 10, 5],
    #     [19, 3, 2],
    #     starts=[20, 10, 4],
    #     ends=[0, 0, 1],
    #     steps=[-1, -3, -2],
    # )


def test_slice_dynamic_inputs_ir():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_dynamic_inputs_ir",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_inputs_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((2,), dtype="int64"),
            ends: R.Tensor((2,), dtype="int64"),
            axes: R.Tensor((2,), dtype="int64"),
            steps: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=3):
            R.func_attr({"num_input": 5})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="bool") = R.less(axes, R.const(0, "int64"))
                lv1: R.Tensor((2,), dtype="int64") = R.add(axes, R.const(3, "int64"))
                lv2: R.Shape([20, 10, 5]) = R.shape_of(x)
                lv3: R.Tensor((2,), dtype="int64") = R.where(lv, lv1, axes)
                lv4: R.Tensor((3,), dtype="int64") = R.shape_to_tensor(lv2)
                lv5: R.Tensor((3,), dtype="int64") = R.scatter_elements(
                    R.const([0, 0, 0], "int64"), lv3, starts, axis=0, reduction="update"
                )
                lv6: R.Tensor((3,), dtype="int64") = R.scatter_elements(
                    lv4, lv3, ends, axis=0, reduction="update"
                )
                lv7: R.Tensor((3,), dtype="int64") = R.scatter_elements(
                    R.const([1, 1, 1], "int64"), lv3, steps, axis=0, reduction="update"
                )
                gv: R.Tensor(dtype="float32", ndim=3) = R.dynamic_strided_slice(x, lv5, lv6, lv7)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_slice_dynamic_inputs_length_validation():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_dynamic_inputs_length_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_inputs_length_validation_test")
    with pytest.raises(ValueError, match="starts and ends to have the same length"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_slice_dynamic_shape_expr_input_validation():
    shape_node = helper.make_node("Shape", ["x"], ["y"])
    slice_node = helper.make_node("Slice", ["y", "starts", "ends", "axes", "steps"], ["z"])

    graph = helper.make_graph(
        [shape_node, slice_node],
        "slice_dynamic_shape_expr_input_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [1]),
        ],
        outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, [1])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_shape_expr_input_validation_test")
    with pytest.raises(ValueError, match="does not support ShapeExpr input"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_slice_zero_step_validation():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_zero_step_validation",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5])],
        initializer=[
            helper.make_tensor("starts", TensorProto.INT64, [2], vals=[0, 0]),
            helper.make_tensor("ends", TensorProto.INT64, [2], vals=[3, 10]),
            helper.make_tensor("axes", TensorProto.INT64, [2], vals=[0, 1]),
            helper.make_tensor("steps", TensorProto.INT64, [2], vals=[1, 0]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_zero_step_validation_test")
    with pytest.raises(ValueError, match="step values must be non-zero"):
        from_onnx(model, opset=13)


def test_slice_dynamic_shape():
    def make_shape_slice_expected(data_shape, starts, ends, axes):
        shape_vars = [
            tvm.tirx.SizeVar(dim, "int64") if isinstance(dim, str) else dim for dim in data_shape
        ]
        x = relax.Var("x", relax.TensorType(shape_vars, "float32"))
        starts_var = relax.Var("starts", relax.TensorType([len(starts)], "int64"))
        ends_var = relax.Var("ends", relax.TensorType([len(ends)], "int64"))
        axes_var = relax.Var("axes", relax.TensorType([len(axes)], "int64"))

        # These test cases slice the 1-D result of Shape(x), so axes=[0].
        assert axes == [0]
        sliced_shape = shape_vars[starts[0] : ends[0]]

        bb = relax.BlockBuilder()
        with bb.function("main", [x, starts_var, ends_var, axes_var]):
            with bb.dataflow():
                if all(isinstance(dim, int) for dim in sliced_shape):
                    out = bb.emit_output(relax.const(sliced_shape, "int64"))
                else:
                    out = bb.emit_output(relax.ShapeExpr(sliced_shape))
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 1)
        return expected

    def verify_slice(
        data_shape, data_instance_shape, output_shape, starts, ends, axes=None, steps=None
    ):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")
        if isinstance(steps, list):
            steps = np.array(steps, "int64")

        slice_inputs = ["y", "starts", "ends"]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
        ]

        if axes is not None:
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes))
            slice_inputs.append("axes")
        if steps is not None:
            initializer.append(helper.make_tensor("steps", TensorProto.INT64, steps.shape, steps))
            slice_inputs.append("steps")

        shape_node = helper.make_node("Shape", inputs=["x"], outputs=["y"])
        slice_node = helper.make_node("Slice", inputs=slice_inputs, outputs=["z"])

        graph = helper.make_graph(
            [shape_node, slice_node],
            "slice_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape),
            ],
            outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, output_shape)],
            initializer=initializer,
        )

        model = helper.make_model(graph, producer_name="slice_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        assert len(tvm_model["main"].attrs["params"]) == 3
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm.ir.assert_structural_equal(
            tvm_model,
            make_shape_slice_expected(data_shape, starts.tolist(), ends.tolist(), axes.tolist()),
        )

    verify_slice([20, 10, 5], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice(["A", 10, 5], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice(["A", "B", 5], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice([20, 10, "C"], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice(["A", "B", "C"], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])

    verify_slice([20, 10, 5], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice(["A", 10, 5], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice(["A", "B", 5], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice([20, 10, "C"], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice(["A", "B", "C"], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])

    verify_slice([20, 10, 5], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice(["A", 10, 5], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice(["A", "B", 5], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice([20, 10, "C"], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice(["A", "B", "C"], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [False])
def test_attention(dynamic):
    def verify_attention(
        input_,
        weight,
        bias,
        mask_index,
        num_heads,
        mask_filter_value,
        qkv_hidden_sizes,
        relative_position_bias,
    ):
        node = onnx.helper.make_node(
            "Attention",
            inputs=["input", "weight", "bias", "mask_index", "", "relative_position_bias"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
            # TODO(jwfromm) OnnxRT doesnt work with this attribute, figure out why not.
            # mask_filter_value=mask_filter_value,
            qkv_hidden_sizes=qkv_hidden_sizes,
        )

        input_shape = list(input_.shape)
        weight_shape = list(weight.shape)
        bias_shape = list(bias.shape)
        mask_shape = list(mask_index.shape)
        relative_position_bias_shape = list(relative_position_bias.shape)
        output_shape = list(input_.shape)
        if dynamic:
            input_shape = ["?" for _ in range(len(input_.shape))]
            weight_shape = ["?" for _ in range(len(weight.shape))]
            bias_shape = ["?" for _ in range(len(bias.shape))]
            mask_shape = ["?" for _ in range(len(mask_index.shape))]
            output_shape = ["?" for _ in range(len(input_.shape))]

        graph = helper.make_graph(
            [node],
            "attention_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("weight", TensorProto.FLOAT, weight_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, mask_shape),
                helper.make_tensor_value_info(
                    "relative_position_bias", TensorProto.FLOAT, relative_position_bias_shape
                ),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="attention_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)

        batch_size, seq_len, input_hidden_size = input_shape
        hidden_size, _, hidden_size_v = qkv_hidden_sizes
        head_size = hidden_size // num_heads
        head_size_v = hidden_size_v // num_heads

        input_var = relax.Var("input", relax.TensorType(input_shape, "float32"))
        weight_var = relax.Var("weight", relax.TensorType(weight_shape, "float32"))
        bias_var = relax.Var("bias", relax.TensorType(bias_shape, "float32"))
        mask_index_var = relax.Var("mask_index", relax.TensorType(mask_shape, "int32"))
        relative_position_bias_var = relax.Var(
            "relative_position_bias",
            relax.TensorType(relative_position_bias_shape, "float32"),
        )

        bb = relax.BlockBuilder()
        with bb.function(
            "main",
            [
                input_var,
                weight_var,
                bias_var,
                mask_index_var,
                relative_position_bias_var,
            ],
        ):
            with bb.dataflow():
                inverted_mask = bb.emit(relax.op.subtract(relax.const(1, "int32"), mask_index_var))
                mask_bias = bb.emit(relax.op.astype(inverted_mask, "float32"))
                mask_bias = bb.emit(relax.op.multiply(mask_bias, relax.const(-10000.0, "float32")))
                mask_bias = bb.emit(relax.op.reshape(mask_bias, [batch_size, 1, 1, seq_len]))
                qkv = bb.emit(relax.op.matmul(input_var, weight_var))
                qkv = bb.emit(relax.op.add(qkv, bias_var))
                qkv_split = bb.emit(relax.op.split(qkv, [hidden_size, hidden_size * 2], axis=2))
                query = bb.emit(qkv_split[0])
                key = bb.emit(qkv_split[1])
                value = bb.emit(qkv_split[2])
                query = bb.emit(
                    relax.op.reshape(query, [batch_size, seq_len, num_heads, head_size])
                )
                key = bb.emit(relax.op.reshape(key, [batch_size, seq_len, num_heads, head_size]))
                value = bb.emit(
                    relax.op.reshape(value, [batch_size, seq_len, num_heads, head_size_v])
                )
                qk_bias = bb.emit(relax.op.add(relative_position_bias_var, mask_bias))
                attention = bb.emit(relax.op.nn.attention(query, key, value, qk_bias))
                output = bb.emit(
                    relax.op.reshape(attention, [batch_size, seq_len, num_heads * head_size_v])
                )
                gv = bb.emit_output(output)
            bb.emit_func_output(gv)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 5)

        tvm.ir.assert_structural_equal(tvm_model, expected)
        # "present" output should be nullptr when the "past" input isn't included,
        # but ort requires an output shape to be specified?
        # verify_with_ort_with_inputs(
        #     model,
        #     [input_, weight, bias, mask_index],
        #     [input_.shape, present_output_shape],
        #     target=target,
        #     dev=dev,
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

    input_hidden_size = 128
    batch_size = 4
    sequence_length = 4
    num_heads = 12
    qkv_hidden_sizes = [192, 192, 96]
    mask_filter_value = -512.0

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, input_hidden_size)).astype(dtype)
    weight = np.random.normal(size=(input_hidden_size, sum(qkv_hidden_sizes))).astype(dtype) * 0.1
    bias = np.random.randn(sum(qkv_hidden_sizes)).astype(dtype)
    mask_index = np.random.randint(2, size=(batch_size, sequence_length)).astype("int32")
    relative_position_bias = np.random.randn(
        batch_size, num_heads, sequence_length, sequence_length
    ).astype(dtype)

    verify_attention(
        input_array,
        weight,
        bias,
        mask_index,
        num_heads,
        mask_filter_value,
        qkv_hidden_sizes,
        relative_position_bias,
    )


def make_pad_expected(input_shape, pads, mode, value, has_pad_inputs):
    data = relax.Var("input", relax.TensorType(list(input_shape), "float32"))
    params = [data]
    if has_pad_inputs:
        params.append(relax.Var("pads", relax.TensorType([len(pads)], "int64")))
        if mode == "constant":
            params.append(relax.Var("constant_value", relax.TensorType([1], "float32")))

    len_dim = len(pads) // 2
    pad_before = list(pads[:len_dim])
    pad_after = list(pads[len_dim:])
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            if mode == "constant":
                out = bb.emit_te(topi.nn.pad, data, pad_before, pad_after, value)
            elif mode == "reflect":
                out = bb.emit_te(topi.nn.mirror_pad, data, pad_before, pad_after, "REFLECT")
            else:
                out = bb.emit_te(topi.nn.replicate_pad, data, pad_before, pad_after)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)
    return expected


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad(dynamic):
    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, mode="constant", value=0.0):
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        pads = np.array(pads)
        #  onnx graph
        if mode in ["edge", "reflect"]:
            outdata = np.pad(np.empty(input_shape, dtype=np.float32), pad_width=np_pads, mode=mode)
            node = helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"], mode=mode)
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            outdata = np.pad(
                np.empty(input_shape, dtype=np.float32),
                pad_width=np_pads,
                mode="constant",
                constant_values=value,
            )
            node = helper.make_node(
                "Pad",
                inputs=["input", "pads", "constant_value"],
                outputs=["output"],
                mode="constant",
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                initializer=[
                    helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                    helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        model.opset_import[0].version = 14
        tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm.ir.assert_structural_equal(
            tvm_model, make_pad_expected(input_shape, pads.tolist(), mode, value, True)
        )

    verify_pad((2, 2), [0, 1, 0, 0], "constant", 0.0)
    verify_pad((2, 3), [1, 0, 0, 1], "constant", 0.0)
    verify_pad((3, 2), [0, 0, 1, 0], "constant", 5.0)
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect")
    verify_pad((2, 3), [1, 1, 1, 1], "edge")
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "edge")


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad_v2(dynamic):
    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, mode="constant", value=0.0):
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        pads = np.array(pads)
        #  onnx graph
        if mode in ["edge", "reflect"]:
            outdata = np.pad(np.empty(input_shape, dtype=np.float32), pad_width=np_pads, mode=mode)
            node = helper.make_node(
                "Pad", inputs=["input"], outputs=["output"], mode=mode, pads=pads
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            outdata = np.pad(
                np.empty(input_shape, dtype=np.float32),
                pad_width=np_pads,
                mode="constant",
                constant_values=value,
            )
            node = helper.make_node(
                "Pad",
                inputs=["input"],
                outputs=["output"],
                mode="constant",
                pads=pads,
                value=value,
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        model.opset_import[0].version = 10
        tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(
            tvm_model, make_pad_expected(input_shape, pads.tolist(), mode, value, False)
        )

    verify_pad((2, 2), [0, 1, 0, 0], "constant", 0.0)
    verify_pad((2, 3), [1, 0, 0, 1], "constant", 0.0)
    verify_pad((3, 2), [0, 0, 1, 0], "constant", 5.0)
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect")
    verify_pad((2, 3), [1, 1, 1, 1], "edge")
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "edge")


@pytest.mark.parametrize("fp_arith", [np.float16, np.float32])
@pytest.mark.parametrize("dynamic", [True, False])
def test_split(fp_arith, dynamic):
    def verify_split(indata_shape, outdata_shapes, split, axis=0, pass_split=True, opset=11):
        indata = np.random.normal(size=indata_shape).astype(fp_arith)
        input_names = ["input"]
        initializer = []

        if split:
            split_index = range(len(split))
        else:
            split_index = range(len(outdata_shapes))

        indata_shape = list(indata.shape)
        if dynamic:
            indata_shape = ["?" for _ in range(len(indata.shape))]
            outdata_shapes = [["?" for _ in range(len(o))] for o in outdata_shapes]

        inputs = [
            helper.make_tensor_value_info(
                "input", helper.np_dtype_to_tensor_dtype(indata.dtype), indata_shape
            )
        ]

        split_constant = None
        if pass_split:
            if opset >= 13:
                np_split = np.array(split).astype(np.int64)
                split_constant = make_constant_node(
                    "split", onnx.TensorProto.INT64, list(np_split.shape), np_split
                )
                input_names.append("split")

        node = helper.make_node(
            "Split",
            inputs=input_names,
            outputs=[f"output_{i}" for i in range(len(split_index))],
            axis=axis,
        )

        if pass_split and opset < 13:
            split_attr = helper.make_attribute("split", split)
            node.attribute.append(split_attr)

        nodes = [split_constant, node] if split_constant else [node]

        graph = helper.make_graph(
            nodes,
            "split_test",
            inputs=inputs,
            initializer=initializer,
            outputs=[
                helper.make_tensor_value_info(
                    f"output_{i}",
                    helper.np_dtype_to_tensor_dtype(indata.dtype),
                    list(outdata_shapes[i]),
                )
                for i in range(len(split_index))
            ],
        )
        model = helper.make_model(graph, producer_name="split_test")
        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

        dtype = str(indata.dtype)
        expected_input_shape = (
            _shape_with_size_vars(indata_shape, "split_input_dim") if dynamic else indata_shape
        )
        data = relax.Var("input", relax.TensorType(expected_input_shape, dtype))

        if pass_split and split and len(split) > 1:
            indices_or_sections = np.cumsum(split[:-1]).astype("int64").tolist()
        else:
            indices_or_sections = len(split_index)

        bb = relax.BlockBuilder()
        with bb.function("main", [data]):
            with bb.dataflow():
                if len(split_index) == 1:
                    out = bb.emit_output(relax.op.split(data, indices_or_sections, axis=axis))
                else:
                    split_outputs = bb.emit(relax.op.split(data, indices_or_sections, axis=axis))
                    outputs = [bb.emit(split_outputs[i]) for i in split_index]
                    out = bb.emit_output(relax.Tuple(outputs))
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 1)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    # 1D
    verify_split(6, [[2], [2], [2]], [2, 2, 2])
    verify_split(6, [[2], [2], [2]], [2, 2, 2], pass_split=False)
    verify_split(6, [[2], [1], [3]], [2, 1, 3])
    verify_split(6, [[2], [1], [3]], [2, 1, 3], opset=13)
    # 2D
    verify_split(
        (4, 4),
        [[2, 2], [2, 2]],
        [2, 2],
        axis=1,
    )
    verify_split(
        (4, 4),
        [[2, 2], [2, 2]],
        [2, 2],
        axis=1,
        opset=13,
    )
    # Split evenly (unstack)
    verify_split(3, [[1], [1], [1]], False, pass_split=False)
    # Split a single value to a single value
    verify_split(1, [[1]], [1], pass_split=True)
    # Test that the default case modifies nothing when split list has length one
    verify_split((1, 2), [[2]], [2], axis=1)
    verify_split((1, 2), [[2]], [1])


@pytest.mark.parametrize("dynamic", [True, False])
def test_tile(dynamic):
    def verify_tile(in_shape, repeats, out_shape):
        node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])

        model_in_shape = list(in_shape)
        model_out_shape = list(out_shape)
        if dynamic:
            model_in_shape = ["?" for _ in range(len(in_shape))]
            model_out_shape = ["?" for _ in range(len(out_shape))]

        graph = helper.make_graph(
            [node],
            "tile_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, model_in_shape),
            ],
            initializer=[
                helper.make_tensor("repeats", TensorProto.INT64, list(repeats.shape), repeats)
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, model_out_shape)],
        )

        model = helper.make_model(
            graph, producer_name="tile_test", opset_imports=[helper.make_opsetid("", 14)]
        )
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_model["main"] = tvm_model["main"].without_attr("params")

        if dynamic:
            expected_input_shape = [
                tvm.tirx.SizeVar(f"tile_input_dim_{i}", "int64") for i in range(len(model_in_shape))
            ]
        else:
            expected_input_shape = list(in_shape)
        data = relax.Var("input", relax.TensorType(expected_input_shape, "float32"))
        repeats_param = relax.Var("repeats", relax.TensorType([len(repeats)], "int64"))
        bb = relax.BlockBuilder()
        with bb.function("main", [data, repeats_param]):
            with bb.dataflow():
                out = bb.emit_te(topi.tile, data, repeats.tolist())
                out = bb.emit_output(out)
            bb.emit_func_output(out)
        expected = bb.get()
        expected["main"] = expected["main"].with_attr("num_input", 1)

        tvm.ir.assert_structural_equal(tvm_model, expected)

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z_array = np.tile(x, repeats)
    verify_tile(x.shape, repeats, z_array.shape)


@pytest.mark.parametrize("dynamic_input", [True, False])
@pytest.mark.parametrize(
    "in_shape,repeats",
    [
        ((2, 3), np.array([2, 2], dtype=np.int64)),
        ((2, 3, 4), np.array([2, 2, 1], dtype=np.int64)),
        ((2, 3, 4, 5), np.array([1, 2, 1, 2], dtype=np.int64)),
    ],
)
def test_tile_dynamic_repeats(dynamic_input, in_shape, repeats):
    out_shape = np.tile(np.empty(in_shape, dtype=np.float32), repeats).shape

    input_shape = ["?" for _ in in_shape] if dynamic_input else list(in_shape)
    output_shape = ["?" for _ in out_shape] if dynamic_input else list(out_shape)

    node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "tile_dynamic_repeats_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("repeats", TensorProto.INT64, [len(repeats)]),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(
        graph,
        producer_name="tile_dynamic_repeats_test",
        opset_imports=[helper.make_opsetid("", 13)],
    )

    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    if dynamic_input:
        expected_input_shape = [
            tvm.tirx.SizeVar(f"tile_data_dim_{i}", "int64") for i in range(len(in_shape))
        ]
    else:
        expected_input_shape = list(in_shape)
    data = relax.Var("input", relax.TensorType(expected_input_shape, "float32"))
    repeats_param = relax.Var("repeats", relax.TensorType([len(repeats)], "int64"))
    bb = relax.BlockBuilder()
    with bb.function("main", [data, repeats_param]):
        with bb.dataflow():
            data_shape = bb.normalize(relax.op.shape_of(data))
            data_shape_tensor = bb.normalize(relax.op.shape_to_tensor(data_shape))
            output_shape_tensor = repeats_param
            data_ndim = len(in_shape)
            repeats_len = len(repeats)
            if data_ndim > repeats_len:
                repeats_prefix = relax.const(
                    np.ones((data_ndim - repeats_len,), dtype="int64"), "int64"
                )
                output_shape_tensor = bb.normalize(
                    relax.op.concat([repeats_prefix, output_shape_tensor], axis=0)
                )
            elif repeats_len > data_ndim:
                data_prefix = relax.const(
                    np.ones((repeats_len - data_ndim,), dtype="int64"), "int64"
                )
                data_shape_tensor = bb.normalize(
                    relax.op.concat([data_prefix, data_shape_tensor], axis=0)
                )

            output_shape_tensor = bb.normalize(
                relax.op.multiply(output_shape_tensor, data_shape_tensor)
            )
            output_shape_expr = bb.normalize(relax.op.tensor_to_shape(output_shape_tensor))
            output_shape_vars = [
                tvm.tirx.Var(f"tile_dim_{i}", "int64") for i in range(max(data_ndim, repeats_len))
            ]
            bb.match_cast(output_shape_expr, relax.ShapeType(output_shape_vars))
            out = bb.emit_te(topi.dyn_tile, data, output_shape_vars, repeats_len)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def _generate_roi_cases():
    # Base case when with_roi is False
    roi_list = [
        pytest.param(False, None, False, id="no_roi"),
    ]

    # Valid when with_roi is True and with_constant is True/False
    roi_cases = [
        [],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.1, 0.1, 0.9, 0.9],
        [0.2, 0.2, 0.8, 0.8],
        [0.3, 0.3, 0.7, 0.7],
        [0.4, 0.4, 0.6, 0.6],
        [0.5, 0.5, 0.5, 0.5],
        [0.1, 0.2, 0.9, 0.8],
    ]
    for roi in roi_cases:
        roi_list.append(pytest.param(True, roi, True, id=f"roi_{'_'.join(str(x) for x in roi)}"))
        roi_list.append(pytest.param(True, roi, False, id=f"roi_{'_'.join(str(x) for x in roi)}"))

    return roi_list


@pytest.mark.parametrize("with_roi, roi_list, with_constant", _generate_roi_cases())
def test_resize(with_roi, roi_list, with_constant):
    nodes = []
    resize_node = helper.make_node(
        "Resize", ["X", "roi" if with_roi else "", "scales"], ["Y"], mode="cubic"
    )

    if with_roi and with_constant:
        roi_tensor = helper.make_tensor(
            name="roi",
            data_type=TensorProto.FLOAT,
            dims=[len(roi_list)],
            vals=roi_list,
        )

        roi_const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["roi"],
            value=roi_tensor,
        )
        nodes.append(roi_const_node)

    nodes.append(resize_node)

    initializers = [
        helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
    ]

    if with_roi and not with_constant:
        initializers.append(helper.make_tensor("roi", TensorProto.FLOAT, [len(roi_list)], roi_list))

    graph = helper.make_graph(
        nodes,
        "resize_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
        initializer=initializers,
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64]),
        ],
    )

    model = helper.make_model(graph, producer_name="resize_test")
    check_correctness(model)


def test_resize_dynamic_roi_tf_crop_and_resize():
    """ROI is a graph input (not initializer), lowered through TOPI dynamic-ROI path."""
    resize_node = helper.make_node(
        "Resize",
        ["X", "roi", "scales"],
        ["Y"],
        mode="linear",
        coordinate_transformation_mode="tf_crop_and_resize",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize_dynamic_roi",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32]),
            helper.make_tensor_value_info("roi", TensorProto.FLOAT, [8]),
        ],
        initializer=[
            helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64]),
        ],
    )
    model = helper.make_model(graph, producer_name="resize_dynamic_roi")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    seen_call_tir = False

    def _visit(expr):
        nonlocal seen_call_tir
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op):
            if expr.op.name == "relax.call_tir":
                seen_call_tir = True

    relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
    assert seen_call_tir


def test_resize_dynamic_roi_3d_tf_crop_and_resize():
    """5-D NCDHW: ROI is a graph input; covers dynamic-ROI TOPI resize3d path."""
    resize_node = helper.make_node(
        "Resize",
        ["X", "roi", "scales"],
        ["Y"],
        mode="linear",
        coordinate_transformation_mode="tf_crop_and_resize",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize_dynamic_roi_3d",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 4, 5]),
            helper.make_tensor_value_info("roi", TensorProto.FLOAT, [10]),
        ],
        initializer=[
            helper.make_tensor("scales", TensorProto.FLOAT, [5], [1.0, 1.0, 2.0, 2.0, 2.0]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 6, 8, 10]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="resize_dynamic_roi_3d",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)
    seen_call_tir = False

    def _visit(expr):
        nonlocal seen_call_tir
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op):
            if expr.op.name == "relax.call_tir":
                seen_call_tir = True

    relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
    assert seen_call_tir


def test_resize_nd_sizes():
    cases = [
        ("resize1d", [1, 1, 4], [1, 1, 7]),
        ("resize2d", [1, 1, 4, 5], [1, 1, 6, 7]),
        ("resize3d", [1, 1, 3, 4, 5], [1, 1, 4, 6, 7]),
    ]

    for name, input_shape, sizes in cases:
        resize_node = helper.make_node(
            "Resize",
            ["X", "", "", "sizes"],
            ["Y"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )

        graph = helper.make_graph(
            [resize_node],
            name,
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
            ],
            initializer=[
                helper.make_tensor("sizes", TensorProto.INT64, [len(sizes)], sizes),
            ],
            outputs=[
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, sizes),
            ],
        )

        model = helper.make_model(
            graph, producer_name=name, opset_imports=[helper.make_opsetid("", 18)]
        )
        if name != "resize1d":
            check_correctness(model, opset=18)
            continue

        tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)
        seen_call_tir = False

        def _visit(expr):
            nonlocal seen_call_tir
            if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op):
                if expr.op.name == "relax.call_tir":
                    seen_call_tir = True

        relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
        assert seen_call_tir


def test_resize_5d_emits_relax_resize3d():
    resize_node = helper.make_node(
        "Resize",
        ["X", "", "", "sizes"],
        ["Y"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize3d_ir_check",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 4, 5])],
        initializer=[helper.make_tensor("sizes", TensorProto.INT64, [5], [1, 1, 4, 6, 7])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 6, 7])],
    )
    model = helper.make_model(graph, producer_name="resize3d_ir_check")
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    seen_resize3d = False

    def _visit(expr):
        nonlocal seen_resize3d
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op):
            if expr.op.name == "relax.image.resize3d":
                seen_resize3d = True

    relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
    assert seen_resize3d


def test_einsum():
    eqn = "ij->i"
    einsum_node = helper.make_node("Einsum", ["x"], ["y"], equation=eqn)

    graph = helper.make_graph(
        [einsum_node],
        "einsum_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [3]),
        ],
    )

    model = helper.make_model(graph, producer_name="einsum_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    x = relax.Var("x", relax.TensorType([3, 4], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            out = bb.emit_te(topi.einsum, eqn, x)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_range():
    range_node = helper.make_node(
        "Range",
        ["start", "limit", "delta"],
        ["output"],
    )

    graph = helper.make_graph(
        [range_node],
        "range_test",
        inputs=[],
        initializer=[
            helper.make_tensor("start", TensorProto.INT64, [], [1]),
            helper.make_tensor("limit", TensorProto.INT64, [], [5]),
            helper.make_tensor("delta", TensorProto.INT64, [], [2]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.INT64, [2]),
        ],
    )

    model = helper.make_model(graph, producer_name="range_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    start = relax.Var("start", relax.TensorType([], "int64"))
    limit = relax.Var("limit", relax.TensorType([], "int64"))
    delta = relax.Var("delta", relax.TensorType([], "int64"))
    bb = relax.BlockBuilder()
    with bb.function("main", [start, limit, delta]):
        with bb.dataflow():
            out = bb.emit_output(relax.const(np.array([1, 3], dtype=np.int64), "int64"))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 0)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_batch_norm():
    batch_norm_node = helper.make_node(
        "BatchNormalization", ["x", "s", "bias", "mean", "var"], ["y"], epsilon=1e-2
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "batch_norm_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5])],
    )

    model = helper.make_model(graph, producer_name="batch_norm_test")
    check_correctness(model, opset=15)


def test_batch_norm_defaults_to_inference_mode():
    batch_norm_node = helper.make_node(
        "BatchNormalization", ["x", "s", "bias", "mean", "var"], ["y"], epsilon=1e-2
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "batch_norm_inference_attr_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5])],
    )
    model = helper.make_model(graph, producer_name="batch_norm_inference_attr_test")
    model.opset_import[0].version = 15

    tvm_model = from_onnx(model, opset=15, keep_params_in_input=True)
    batch_norm_attrs = []

    def visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm.ir.Op.get("relax.nn.batch_norm"):
            batch_norm_attrs.append(expr.attrs)

    relax.analysis.post_order_visit(tvm_model["main"], visit)

    assert len(batch_norm_attrs) == 1
    assert batch_norm_attrs[0].training is False


def get_pool_padding(shape, auto_pad, kernel_shape, strides, pads):
    def get_pad_pair(input1d, kernel1d, stride1d, mode):
        if input1d % stride1d == 0:
            pad = max(kernel1d - stride1d, 0)
        else:
            pad = max(kernel1d - (input1d % stride1d), 0)
        pad_before = pad // 2
        pad_after = pad - pad_before
        if "LOWER" in mode:
            return [pad_after, pad_before]
        return [pad_before, pad_after]

    strides = strides or [1] * (len(shape) - 2)
    padding = pads if pads is not None else 0

    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        pad_pairs = [
            get_pad_pair(int(shape[2 + axis]), kernel_shape[axis], strides[axis], auto_pad)
            for axis in range(len(shape) - 2)
        ]
        padding = tuple(val for pair in zip(*pad_pairs) for val in pair)

    return padding


def make_pool_expected(pool_name, shape, auto_pad, kernel_shape, strides, pads):
    data = relax.Var("x", relax.TensorType(shape, "float32"))
    strides = strides or [1] * (len(shape) - 2)
    dilations = [1] * (len(shape) - 2)
    padding = get_pool_padding(shape, auto_pad, kernel_shape, strides, pads)
    pool_kind = "max" if pool_name == "MaxPool" else "avg"
    pool_op = getattr(relax.op.nn, pool_kind + "_pool" + str(len(kernel_shape)) + "d")

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow():
            output = bb.emit_output(
                pool_op(data, kernel_shape, strides, padding, dilations, False, False)
            )
        bb.emit_func_output(output)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)
    return expected


def make_lppool_expected(shape, auto_pad, kernel_shape, strides, pads):
    data = relax.Var("x", relax.TensorType(shape, "float32"))
    strides = strides or [1] * (len(shape) - 2)
    dilations = [1] * (len(shape) - 2)
    padding = get_pool_padding(shape, auto_pad, kernel_shape, strides, pads)
    pool_op = getattr(relax.op.nn, "avg_pool" + str(len(kernel_shape)) + "d")

    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow():
            powered = bb.emit(relax.op.power(data, relax.const(2.0, dtype="float32")))
            pooled = bb.emit(pool_op(powered, kernel_shape, strides, padding, dilations, 0, True))
            scaled = bb.emit(
                relax.op.multiply(pooled, relax.const(float(np.prod(kernel_shape)), "float32"))
            )
            output = bb.emit_output(relax.op.power(scaled, relax.const(0.5, dtype="float32")))
        bb.emit_func_output(output)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)
    return expected


def verify_pool_ir(pool_name, shape, auto_pad, kernel_shape, strides, pads):
    attrs = {
        "kernel_shape": kernel_shape,
        "strides": strides,
        "auto_pad": auto_pad,
    }
    if pads is not None:
        attrs["pads"] = pads

    node = helper.make_node(pool_name, ["x"], ["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "pool_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, producer_name="pool_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    if pool_name == "LpPool":
        expected = make_lppool_expected(shape, auto_pad, kernel_shape, strides, pads)
    else:
        expected = make_pool_expected(pool_name, shape, auto_pad, kernel_shape, strides, pads)
    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("pool_name", ["MaxPool", "AveragePool", "LpPool"])
@pytest.mark.parametrize(
    "shape, auto_pad, kernel_shape, strides, pads",
    [
        # Pool1D
        ([1, 1, 32], "NOTSET", [3], [1], [1, 1]),
        # Pool1D with stride
        ([1, 1, 32], "NOTSET", [3], [2], [1, 1]),
        # Pool1D with stride and autopadding
        ([1, 1, 32], "SAME_UPPER", [7], [2], None),
        ([1, 1, 32], "SAME_LOWER", [4], [4], None),
        ([1, 1, 32], "VALID", [5], [5], None),
        ([1, 1, 32], "SAME_UPPER", [3], [1], None),
        # Pool2D
        ([1, 1, 32, 32], "NOTSET", [3, 3], [1, 1], [1, 1, 1, 1]),
        # Pool2D with stride
        ([1, 1, 32, 32], "NOTSET", [3, 3], [2, 2], [1, 1, 1, 1]),
        # Pool2D with stride and autopadding
        ([1, 1, 32, 32], "SAME_UPPER", [3, 7], [3, 2], None),
        ([1, 1, 32, 32], "SAME_LOWER", [3, 3], [2, 2], None),
        ([1, 1, 32, 32], "VALID", [3, 3], [2, 2], None),
        ([1, 1, 32, 32], "SAME_UPPER", [3, 3], [1, 1], None),
        # Pool3D
        ([1, 1, 32, 32, 32], "NOTSET", [3, 3, 4], [1, 1, 1], [1, 2, 1, 1, 2, 2]),
        # Pool3D with stride
        ([1, 1, 32, 32, 32], "NOTSET", [3, 4, 3], [2, 2, 3], [1, 1, 1, 1, 1, 2]),
        # Pool3D with stride and autopadding
        ([1, 1, 32, 32, 32], "SAME_UPPER", [4, 3, 3], [3, 2, 2], None),
        ([1, 1, 32, 32, 32], "SAME_LOWER", [3, 3, 4], [2, 2, 2], None),
        ([1, 1, 32, 32, 32], "VALID", [3, 3, 5], [2, 2, 3], None),
        ([1, 1, 32, 32, 32], "SAME_UPPER", [3, 3, 5], [1, 1, 1], None),
    ],
)
def test_pool(
    pool_name: str,
    shape: list[int],
    auto_pad: str,
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
):
    verify_pool_ir(pool_name, shape, auto_pad, kernel_shape, strides, pads)


def test_global_average_pool():
    def verify_global_average_pool_ir(input_shape, expected):
        output_shape = input_shape[:2] + [1] * (len(input_shape) - 2)
        node = helper.make_node("GlobalAveragePool", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "global_average_pool_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="global_average_pool_test",
            opset_imports=[helper.make_opsetid("", 14)],
        )
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class Expected1D:
        @R.function
        def main(x: R.Tensor((1, 3, 32), dtype="float32")) -> R.Tensor((1, 3, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1), dtype="float32") = R.mean(x, axis=[2], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(x, axis=[2, 3], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.mean(
                    x, axis=[2, 3, 4], keepdims=True
                )
                R.output(gv)
            return gv

    verify_global_average_pool_ir([1, 3, 32], Expected1D)
    verify_global_average_pool_ir([1, 3, 32, 32], Expected2D)
    verify_global_average_pool_ir([1, 3, 32, 32, 32], Expected3D)


def test_global_max_pool():
    def verify_global_max_pool_ir(input_shape, expected):
        output_shape = input_shape[:2] + [1] * (len(input_shape) - 2)
        node = helper.make_node("GlobalMaxPool", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "global_max_pool_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="global_max_pool_test",
            opset_imports=[helper.make_opsetid("", 14)],
        )
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class Expected1D:
        @R.function
        def main(x: R.Tensor((1, 3, 32), dtype="float32")) -> R.Tensor((1, 3, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1), dtype="float32") = R.max(x, axis=[2], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.max(x, axis=[2, 3], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.max(
                    x, axis=[2, 3, 4], keepdims=True
                )
                R.output(gv)
            return gv

    verify_global_max_pool_ir([1, 3, 32], Expected1D)
    verify_global_max_pool_ir([1, 3, 32, 32], Expected2D)
    verify_global_max_pool_ir([1, 3, 32, 32, 32], Expected3D)


def make_global_lp_pool_expected(input_shape: list[int], p: int):
    output_shape = input_shape[:2] + [1] * (len(input_shape) - 2)
    axes = list(range(2, len(input_shape)))

    x = relax.Var("x", relax.TensorType(input_shape, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            abs_x = bb.emit(relax.op.abs(x))
            powered = bb.emit(relax.op.power(abs_x, relax.const(float(p), "float32")))
            summed = bb.emit(relax.op.sum(powered, axes, keepdims=True))
            out = bb.emit_output(relax.op.power(summed, relax.const(float(1 / p), "float32")))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)
    return expected


@pytest.mark.parametrize("p", [1, 2, 3])
def test_global_lp_pool(p: int):
    for input_shape in ([1, 3, 4], [1, 3, 4, 4], [1, 3, 4, 4, 4]):
        output_shape = input_shape[:2] + [1] * (len(input_shape) - 2)
        node = helper.make_node("GlobalLpPool", ["x"], ["y"], p=p)
        graph = helper.make_graph(
            [node],
            "global_lp_pool_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(graph, producer_name="global_lp_pool_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, make_global_lp_pool_expected(input_shape, p))


def make_maxunpool_expected(input_shape, kernel_shape, pads, strides):
    data = relax.Var("X", relax.TensorType(input_shape, "float32"))
    indices = relax.Var("I_1", relax.TensorType(input_shape, "int64"))
    strides = strides or [1] * len(kernel_shape)

    output_shape = np.concatenate([[1, 1], list(strides)]) * np.array(input_shape)
    output_shape += np.concatenate([[0, 0], list(kernel_shape)], axis=0)
    output_shape -= np.concatenate([[0, 0], list(strides)], axis=0)

    if pads is not None:
        pads_by_axis = np.concatenate([[0, 0, 0, 0], list(pads)], axis=0).reshape([-1, 2])
        output_shape -= np.sum(pads_by_axis, axis=-1)

    output_shape = [int(dim) for dim in output_shape.tolist()]
    output_shape_expr = relax.ShapeExpr(output_shape)

    bb = relax.BlockBuilder()
    with bb.function("main", [data, indices]):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros(output_shape_expr, data.ty.dtype))
            flat_zeros = bb.emit(relax.op.reshape(zeros, [-1]))
            flat_indices = bb.emit(relax.op.reshape(indices, [-1]))
            flat_data = bb.emit(relax.op.reshape(data, [-1]))
            scattered = bb.emit(
                relax.op.scatter_elements(flat_zeros, flat_indices, flat_data, axis=0)
            )
            output = bb.emit_output(relax.op.reshape(scattered, output_shape_expr))
        bb.emit_func_output(output)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)
    return expected


@pytest.mark.parametrize("kernel_shape", [[2, 2], [3, 3]])
@pytest.mark.parametrize("pads", [None, [1, 1, 1, 1]])
@pytest.mark.parametrize("strides", [None, [2, 2]])
def test_maxunpool(kernel_shape, pads, strides):
    input_shape = [16, 3, 16, 16]
    input_names = ["X", "I"]
    input_info = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
        helper.make_tensor_value_info("I", TensorProto.INT64, input_shape),
    ]

    attrs = {"kernel_shape": kernel_shape}
    if pads is not None:
        attrs["pads"] = pads
    if strides is not None:
        attrs["strides"] = strides

    node = helper.make_node("MaxUnpool", inputs=input_names, outputs=["y"], **attrs)

    graph = helper.make_graph(
        [node],
        "maxunpool_test",
        inputs=input_info,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )

    model = helper.make_model(graph, producer_name="maxunpool_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    tvm.ir.assert_structural_equal(
        tvm_model, make_maxunpool_expected(input_shape, kernel_shape, pads, strides)
    )


def test_dropout():
    def verify_dropout_ir(opset, attrs, expected):
        node = helper.make_node("Dropout", ["x"], ["y"], **attrs)
        graph = helper.make_graph(
            [node],
            "dropout_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 32, 32])],
        )
        model = helper.make_model(
            graph,
            producer_name="dropout_structural_test",
            opset_imports=[helper.make_opsetid("", opset)],
        )
        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedDropoutRateHalf:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 32, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 32, 32), dtype="float32"),
                    R.Tensor((1, 3, 32, 32), dtype="float32"),
                ) = R.nn.dropout(x, rate=0.5)
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 3, 32, 32), dtype="float32") = lv[1]
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_dropout_ir(14, {}, ExpectedDropoutRateHalf)
    verify_dropout_ir(11, {"ratio": 0.5}, ExpectedDropoutRateHalf)

    # Opset 12+ passes ratio as an optional input; check it is captured into the relax op.
    node = helper.make_node("Dropout", ["x", "ratio"], ["y"])
    graph = helper.make_graph(
        [node],
        "dropout_ratio_input",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        initializer=[helper.make_tensor("ratio", TensorProto.FLOAT, [], [0.3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    model = helper.make_model(graph, producer_name="dropout_ratio_input")
    model.opset_import[0].version = 13
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=False)

    @I.ir_module
    class ExpectedDropoutRatioInput:
        @R.function
        def main(x: R.Tensor((1, 3, 4, 4), dtype="float32")) -> R.Tensor(
            (1, 3, 4, 4), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 4, 4), dtype="float32"),
                    R.Tensor((1, 3, 4, 4), dtype="float32"),
                ) = R.nn.dropout(x, rate=0.30000001192092896)
                lv1: R.Tensor((1, 3, 4, 4), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 3, 4, 4), dtype="float32") = lv[1]
                gv: R.Tensor((1, 3, 4, 4), dtype="float32") = lv1
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, ExpectedDropoutRatioInput)


def test_flatten():
    def verify_flatten_ir(axis, output_shape, expected):
        node = helper.make_node("Flatten", ["x"], ["y"], axis=axis)
        graph = helper.make_graph(
            [node],
            "flatten_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(graph, producer_name="flatten_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedAxis0:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3072), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3072), dtype="float32") = R.reshape(x, R.shape([1, 3072]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedAxisNegative1:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (96, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((96, 32), dtype="float32") = R.reshape(x, R.shape([96, 32]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedAxis2:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (3, 1024), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 1024), dtype="float32") = R.reshape(x, R.shape([3, 1024]))
                R.output(gv)
            return gv

    verify_flatten_ir(0, [1, 3072], ExpectedAxis0)
    verify_flatten_ir(-1, [96, 32], ExpectedAxisNegative1)
    verify_flatten_ir(2, [3, 1024], ExpectedAxis2)


def test_flatten_dynamic():
    def verify_flatten_dynamic_ir(axis, expected):
        node = helper.make_node("Flatten", ["x"], ["y"], axis=axis)
        graph = helper.make_graph(
            [node],
            "flatten_dynamic_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "A", "B", 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None])],
        )
        model = helper.make_model(graph, producer_name="flatten_dynamic_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedDynamicAxis0:
        @R.function
        def main(x: R.Tensor((1, "A", "B", 32), dtype="float32")) -> R.Tensor(
            (1, "A * B * 32"), dtype="float32"
        ):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, A * B * 32), dtype="float32") = R.reshape(
                    x, R.shape([1, A * B * 32])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedDynamicAxisNegative1:
        @R.function
        def main(x: R.Tensor((1, "A", "B", 32), dtype="float32")) -> R.Tensor(
            ("A * B", 32), dtype="float32"
        ):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A * B, 32), dtype="float32") = R.reshape(x, R.shape([A * B, 32]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedDynamicAxis2:
        @R.function
        def main(x: R.Tensor((1, "A", "B", 32), dtype="float32")) -> R.Tensor(
            ("A", "B * 32"), dtype="float32"
        ):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B * 32), dtype="float32") = R.reshape(x, R.shape([A, B * 32]))
                R.output(gv)
            return gv

    verify_flatten_dynamic_ir(0, ExpectedDynamicAxis0)
    verify_flatten_dynamic_ir(-1, ExpectedDynamicAxisNegative1)
    verify_flatten_dynamic_ir(2, ExpectedDynamicAxis2)


def test_onehot():
    one_hot_node = helper.make_node("OneHot", ["indices", "depth", "values"], ["y"], axis=1)
    graph = helper.make_graph(
        [one_hot_node],
        "one_hot_test",
        inputs=[
            helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2]),
        ],
        initializer=[
            helper.make_tensor("depth", TensorProto.INT64, [], [10]),
            helper.make_tensor("values", TensorProto.FLOAT, [2], [3, 1]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 10, 2])],
    )

    model = helper.make_model(graph, producer_name="one_hot_test")
    values = {
        "indices": np.array([[1, 9], [2, 4]], dtype="int64"),
    }
    check_correctness(model, inputs=values)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("sorted", [0, 1])
@pytest.mark.parametrize("num_outputs", [1, 2, 3, 4])
def test_unique(axis: int | None, sorted: int, num_outputs: int):
    if num_outputs in [3, 4] and axis is None:
        pytest.xfail("RuntimeError: Check failed: input_shape.size() == size (2 vs. 1)")

    input_shape = [8, 8]
    if axis is None:
        output_shape = [-1]
    else:
        output_shape = [8, 8]
        output_shape[axis] = -1

    output_names = ["y", "indices", "inverse_indices", "counts"][:num_outputs]
    unique_node = helper.make_node("Unique", ["x"], output_names, axis=axis, sorted=sorted)

    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)]
    if num_outputs > 1:
        outputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, [-1]))
    if num_outputs > 2:
        # ONNX spec: inverse_indices is always 1D
        outputs.append(helper.make_tensor_value_info("inverse_indices", TensorProto.INT64, [-1]))
    if num_outputs > 3:
        outputs.append(helper.make_tensor_value_info("counts", TensorProto.INT64, [-1]))

    graph = helper.make_graph(
        [unique_node],
        "unique_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=outputs,
    )
    model = helper.make_model(graph, producer_name="unique_test")
    check_correctness(model)


@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (4, 5, 6), (7, 8, 9, 10)])
def test_nonzero(shape):
    ndim = max(len(shape), 1)
    node = helper.make_node("NonZero", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "nonzero_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.BOOL, shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, [ndim, None])],
    )
    model = helper.make_model(graph, producer_name="nonzero_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    nonzero_numbers = tvm.tirx.Var("nonzero_numbers", "int64")
    x = relax.Var("x", relax.TensorType(shape, "bool"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        with bb.dataflow():
            nonzero = bb.match_cast(
                relax.op.nonzero(x),
                relax.TensorType([ndim, nonzero_numbers], "int64"),
            )
            out = bb.emit_output(nonzero)
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("mode", ["DCR", "CRD"])
def test_depth_to_space(mode: Literal["DCR", "CRD"]):
    in_shape = [1, 8, 2, 3]
    out_shape = [1, 2, 4, 6]
    blocksize = 2
    node = onnx.helper.make_node(
        "DepthToSpace", inputs=["x"], outputs=["y"], blocksize=blocksize, mode=mode
    )
    graph = helper.make_graph(
        [node],
        "depth_to_space_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="depth_to_space_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if mode == "DCR":

        @I.ir_module
        class ExpectedDCR:
            @R.function
            def main(
                x: R.Tensor((1, 8, 2, 3), dtype="float32"),
            ) -> R.Tensor((1, 2, 4, 6), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((1, 2, 2, 2, 2, 3), dtype="float32") = R.reshape(
                        x, R.shape([1, 2, 2, 2, 2, 3])
                    )
                    lv1: R.Tensor((1, 2, 2, 2, 3, 2), dtype="float32") = R.permute_dims(
                        lv, axes=[0, 3, 4, 1, 5, 2]
                    )
                    gv: R.Tensor((1, 2, 4, 6), dtype="float32") = R.reshape(
                        lv1, R.shape([1, 2, 4, 6])
                    )
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedDCR)

    else:

        @I.ir_module
        class ExpectedCRD:
            @R.function
            def main(
                x: R.Tensor((1, 8, 2, 3), dtype="float32"),
            ) -> R.Tensor((1, 2, 4, 6), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((1, 2, 2, 2, 2, 3), dtype="float32") = R.reshape(
                        x, R.shape([1, 2, 2, 2, 2, 3])
                    )
                    lv1: R.Tensor((1, 2, 2, 2, 3, 2), dtype="float32") = R.permute_dims(
                        lv, axes=[0, 1, 4, 2, 5, 3]
                    )
                    gv: R.Tensor((1, 2, 4, 6), dtype="float32") = R.reshape(
                        lv1, R.shape([1, 2, 4, 6])
                    )
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedCRD)


def test_space_to_depth():
    in_shape = [1, 2, 4, 6]
    out_shape = [1, 8, 2, 3]
    blocksize = 2
    node = onnx.helper.make_node("SpaceToDepth", inputs=["x"], outputs=["y"], blocksize=blocksize)
    graph = helper.make_graph(
        [node],
        "space_to_depth_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="space_to_depth_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 2, 4, 6), dtype="float32"),
        ) -> R.Tensor((1, 8, 2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 3, 2), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 3, 2])
                )
                lv1: R.Tensor((1, 2, 2, 2, 2, 3), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 3, 5, 1, 2, 4]
                )
                gv: R.Tensor((1, 8, 2, 3), dtype="float32") = R.reshape(lv1, R.shape([1, 8, 2, 3]))
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def construct_sequence(input_shape: list[int], num_tensors: int, name: str = "sequence"):
    inputs = [f"data{i}" for i in range(num_tensors)]
    sequence_construct_node = helper.make_node("SequenceConstruct", inputs, [name])
    graph_inputs = [
        helper.make_tensor_value_info(f"data{i}", TensorProto.FLOAT, input_shape)
        for i in range(num_tensors)
    ]
    return sequence_construct_node, graph_inputs


def make_constant_node(name: str, data_type: int, dims: list[int], vals: list[int]):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=vals),
    )


def make_optional_tensor_value_info(name: str, elem_type: int, shape: list[int]):
    return helper.make_value_info(
        name, helper.make_optional_type_proto(helper.make_tensor_type_proto(elem_type, shape))
    )


def make_optional_sequence_value_info(name: str, elem_type: int, shape: list[int]):
    return helper.make_value_info(
        name,
        helper.make_optional_type_proto(
            helper.make_sequence_type_proto(helper.make_tensor_type_proto(elem_type, shape))
        ),
    )


def test_sequence_construct():
    node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=2)
    graph = helper.make_graph(
        [node],
        "test_sequence_construct",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_construct")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 32), dtype="float32"), R.Tensor((32, 32), dtype="float32")):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                ) = data0, data1
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_sequence_empty():
    sequence_empty_node = helper.make_node("SequenceEmpty", [], ["sequence"])
    graph = helper.make_graph(
        [sequence_empty_node],
        "test_sequence_empty",
        inputs=[],
        outputs=[helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_empty")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tuple:
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tuple = R.tuple()
                R.output(gv)
            return R.tuple()

    tvm.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("explicit_position", [True, False])
def test_sequence_erase(explicit_position: bool):
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    node_input = ["sequence", "index"] if explicit_position else ["sequence"]
    sequence_erase_node = helper.make_node("SequenceErase", node_input, ["output"])
    graph = helper.make_graph(
        [index, seq_node, sequence_erase_node],
        "test_sequence_erase",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_erase")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if explicit_position:

        @I.ir_module
        class ExpectedEraseExplicit:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
                data2: R.Tensor((32, 32), dtype="float32"),
                data3: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
            ):
                R.func_attr({"num_input": 4})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                    ) = data0, data2, data3
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedEraseExplicit)

    else:

        @I.ir_module
        class ExpectedEraseDefault:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
                data2: R.Tensor((32, 32), dtype="float32"),
                data3: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
            ):
                R.func_attr({"num_input": 4})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                    ) = data0, data1, data2
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedEraseDefault)


@pytest.mark.parametrize("explicit_position", [True, False])
def test_sequence_insert(explicit_position: bool):
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [0])
    node_input = ["sequence", "value", "index"] if explicit_position else ["sequence", "value"]
    sequence_insert_node = helper.make_node("SequenceInsert", node_input, ["output"])
    graph = helper.make_graph(
        [index, seq_node, sequence_insert_node],
        "test_sequence_insert",
        inputs=[*graph_inputs, helper.make_tensor_value_info("value", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_insert")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if explicit_position:

        @I.ir_module
        class ExpectedInsertExplicit:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
                data2: R.Tensor((32, 32), dtype="float32"),
                data3: R.Tensor((32, 32), dtype="float32"),
                value: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
            ):
                R.func_attr({"num_input": 5})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                    ) = value, data0, data1, data2, data3
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedInsertExplicit)

    else:

        @I.ir_module
        class ExpectedInsertDefault:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
                data2: R.Tensor((32, 32), dtype="float32"),
                data3: R.Tensor((32, 32), dtype="float32"),
                value: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
                R.Tensor((32, 32), dtype="float32"),
            ):
                R.func_attr({"num_input": 5})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                        R.Tensor((32, 32), dtype="float32"),
                    ) = data0, data1, data2, data3, value
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedInsertDefault)


@pytest.mark.parametrize(
    "new_axis,axis,expected_shape",
    [
        (0, 0, [64, 32]),
        (0, 1, [32, 64]),
        (1, 0, [2, 32, 32]),
        (1, 1, [32, 2, 32]),
        (1, -1, [32, 32, 2]),
    ],
)
def test_concat_from_sequence(new_axis: int, axis: int, expected_shape: list[int]):
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=2)
    concat_from_sequence_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=axis, new_axis=new_axis
    )
    graph = helper.make_graph(
        [seq_node, concat_from_sequence_node],
        "test_concat_from_sequence",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, expected_shape)],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if new_axis == 0 and axis == 0:

        @I.ir_module
        class ExpectedConcatAxis0:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tensor((64, 32), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    gv: R.Tensor((64, 32), dtype="float32") = R.concat((data0, data1), axis=0)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedConcatAxis0)
    elif new_axis == 0 and axis == 1:

        @I.ir_module
        class ExpectedConcatAxis1:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tensor((32, 64), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    gv: R.Tensor((32, 64), dtype="float32") = R.concat((data0, data1), axis=1)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedConcatAxis1)
    elif new_axis == 1 and axis == 0:

        @I.ir_module
        class ExpectedStackAxis0:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tensor((2, 32, 32), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Tensor((1, 32, 32), dtype="float32") = R.expand_dims(data0, axis=[0])
                    lv1: R.Tensor((1, 32, 32), dtype="float32") = R.expand_dims(data1, axis=[0])
                    gv: R.Tensor((2, 32, 32), dtype="float32") = R.concat((lv, lv1), axis=0)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedStackAxis0)
    elif new_axis == 1 and axis == 1:

        @I.ir_module
        class ExpectedStackAxis1:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tensor((32, 2, 32), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Tensor((32, 1, 32), dtype="float32") = R.expand_dims(data0, axis=[1])
                    lv1: R.Tensor((32, 1, 32), dtype="float32") = R.expand_dims(data1, axis=[1])
                    gv: R.Tensor((32, 2, 32), dtype="float32") = R.concat((lv, lv1), axis=1)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedStackAxis1)
    else:

        @I.ir_module
        class ExpectedStackAxisMinusOne:
            @R.function
            def main(
                data0: R.Tensor((32, 32), dtype="float32"),
                data1: R.Tensor((32, 32), dtype="float32"),
            ) -> R.Tensor((32, 32, 2), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Tensor((32, 32, 1), dtype="float32") = R.expand_dims(data0, axis=[-1])
                    lv1: R.Tensor((32, 32, 1), dtype="float32") = R.expand_dims(data1, axis=[-1])
                    gv: R.Tensor((32, 32, 2), dtype="float32") = R.concat((lv, lv1), axis=-1)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedStackAxisMinusOne)


def test_concat_from_sequence_new_axis_three_tensors():
    """new_axis=1 with three sequence elements (stack then concat along axis)."""
    seq_node, graph_inputs = construct_sequence(input_shape=[16, 8], num_tensors=3)
    concat_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=0, new_axis=1
    )
    graph = helper.make_graph(
        [seq_node, concat_node],
        "test_concat_from_sequence_new_axis_three",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 16, 8])],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence_new_axis_three")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((16, 8), dtype="float32"),
            data1: R.Tensor((16, 8), dtype="float32"),
            data2: R.Tensor((16, 8), dtype="float32"),
        ) -> R.Tensor((3, 16, 8), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 16, 8), dtype="float32") = R.expand_dims(data0, axis=[0])
                lv1: R.Tensor((1, 16, 8), dtype="float32") = R.expand_dims(data1, axis=[0])
                lv2: R.Tensor((1, 16, 8), dtype="float32") = R.expand_dims(data2, axis=[0])
                gv: R.Tensor((3, 16, 8), dtype="float32") = R.concat((lv, lv1, lv2), axis=0)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_concat_from_sequence_invalid_new_axis():
    """Verify that new_axis values other than 0 or 1 raise a ValueError."""
    seq_node, graph_inputs = construct_sequence(input_shape=[16, 8], num_tensors=2)
    concat_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=0, new_axis=2
    )
    graph = helper.make_graph(
        [seq_node, concat_node],
        "test_concat_from_sequence_invalid_new_axis",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 8])],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence_invalid_new_axis")

    with pytest.raises(ValueError, match="ConcatFromSequence only supports new_axis in"):
        from_onnx(model, opset=11)


@pytest.mark.parametrize(
    "split,data_shape,output_shape",
    [
        (2, [6, 32], [2, 32]),
        ([16, 48], [64, 32], [32, 32]),
    ],
)
def test_split_to_sequence(split, data_shape: list[int], output_shape: list[int]):
    split_to_sequence_node = helper.make_node(
        "SplitToSequence",
        ["data", "split"],
        ["output"],
        axis=0,
    )
    split_shape = [len(split)] if isinstance(split, list) else ()
    split_node = make_constant_node(
        "split", TensorProto.INT64, split_shape, [split] if isinstance(split, int) else split
    )
    graph = helper.make_graph(
        [split_node, split_to_sequence_node],
        "test_split_to_sequence",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="test_split_to_sequence")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if isinstance(split, int):

        @I.ir_module
        class ExpectedScalarSplit:
            @R.function
            def main(
                data: R.Tensor((6, 32), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((2, 32), dtype="float32"),
                R.Tensor((2, 32), dtype="float32"),
                R.Tensor((2, 32), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((2, 32), dtype="float32"),
                        R.Tensor((2, 32), dtype="float32"),
                        R.Tensor((2, 32), dtype="float32"),
                    ) = R.split(data, indices_or_sections=3, axis=0)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedScalarSplit)

    else:

        @I.ir_module
        class ExpectedSectionsSplit:
            @R.function
            def main(
                data: R.Tensor((64, 32), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((16, 32), dtype="float32"),
                R.Tensor((48, 32), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((16, 32), dtype="float32"),
                        R.Tensor((48, 32), dtype="float32"),
                    ) = R.split(data, indices_or_sections=[16], axis=0)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedSectionsSplit)


def test_sequence_at():
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    node_input = ["sequence", "index"]
    sequence_at_node = helper.make_node("SequenceAt", node_input, ["output"])
    graph = helper.make_graph(
        [index, seq_node, sequence_at_node],
        "test_sequence_at",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_at")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((32, 32), dtype="float32"):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = data1
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_get_element_tensor():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_tensor",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_tensor")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = x
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_has_element_tensor():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_tensor",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_tensor")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.const(True, "bool")
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_has_element_empty():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_empty",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_empty")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), dtype="bool"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.const(False, "bool")
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_has_element_empty_ir():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_empty_ir",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_empty_ir")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), dtype="bool"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.const(False, "bool")
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_get_element_tensor_ir():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_tensor_ir",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_tensor_ir")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = x
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_get_element_sequence():
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    optional_node = helper.make_node("Optional", ["sequence"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["unwrapped"])
    sequence_at_node = helper.make_node("SequenceAt", ["unwrapped", "index"], ["output"])
    graph = helper.make_graph(
        [index, seq_node, optional_node, get_element_node, sequence_at_node],
        "test_optional_get_element_sequence",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])],
        value_info=[make_optional_sequence_value_info("optional", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_sequence")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((32, 32), dtype="float32"):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = data1
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_without_input_requires_type_attr():
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 3])
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"])
    graph = helper.make_graph(
        [optional_node],
        "test_optional_without_input_requires_type_attr",
        inputs=[],
        outputs=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_without_input_requires_type_attr")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="type attribute"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_empty_optional_graph_output_raises():
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 3])
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    graph = helper.make_graph(
        [optional_node],
        "test_empty_optional_graph_output_raises",
        inputs=[],
        outputs=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_empty_optional_graph_output_raises")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="Empty optional graph outputs are not supported"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_optional_has_element_requires_one_input():
    has_element_node = helper.make_node("OptionalHasElement", [], ["output"])
    graph = helper.make_graph(
        [has_element_node],
        "test_optional_has_element_requires_one_input",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_requires_one_input")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="expects one input"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_optional_get_element_empty_raises():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_empty_raises",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_empty_raises")
    model.opset_import[0].version = 18
    with pytest.raises(ValueError, match="empty optional"):
        from_onnx(model, opset=18, keep_params_in_input=True)


@pytest.mark.parametrize("with_reshape_flatten", [False, True])
def test_symbolic_shape_deduction(with_reshape_flatten):
    index_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["indices"],
        value=helper.make_tensor("indices", TensorProto.INT64, [], [0]),
    )
    shape_node = helper.make_node("Shape", ["data"], ["shape_output"])
    nodes = [index_node, shape_node]
    gather_input = "shape_output"

    if with_reshape_flatten:
        reshape_node = helper.make_node(
            "Reshape", ["shape_output", "target_shape"], ["reshaped_shape"]
        )
        nodes.append(reshape_node)
        gather_input = "reshaped_shape"

    gather_node = helper.make_node("Gather", [gather_input, "indices"], ["gather_output"])
    unsqueeze_node = helper.make_node("Unsqueeze", ["gather_output", "axes"], ["unsqueeze_output"])
    constant_of_shape_node = helper.make_node(
        "ConstantOfShape",
        ["unsqueeze_output"],
        ["output"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [1]),
    )
    nodes.extend([gather_node, unsqueeze_node, constant_of_shape_node])

    initializers = [helper.make_tensor("axes", TensorProto.INT64, [1], vals=[0])]
    if with_reshape_flatten:
        initializers.append(helper.make_tensor("target_shape", TensorProto.INT64, [1], vals=[-1]))

    graph = helper.make_graph(
        nodes,
        "test_shape_deduction",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, ["batch", "seq"]),
        ],
        initializer=initializers,
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [1])],
    )
    model = helper.make_model(graph, producer_name="test_shape_deduction")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if with_reshape_flatten:

        @I.ir_module
        class ExpectedWithReshapeFlatten:
            @R.function
            def main(
                data: R.Tensor(("batch", "seq"), dtype="float32"),
                axes: R.Tensor((1,), dtype="int64"),
                target_shape: R.Tensor((1,), dtype="int64"),
            ) -> R.Tensor(("batch",), dtype="float32"):
                batch = T.int64(is_size_var=True)
                seq = T.int64(is_size_var=True)
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((batch,), dtype="float32") = R.broadcast_to(
                        R.const(1, "float32"), R.shape([batch])
                    )
                    R.output(gv)
                return gv

        expected = ExpectedWithReshapeFlatten

    else:

        @I.ir_module
        class ExpectedWithoutReshapeFlatten:
            @R.function
            def main(
                data: R.Tensor(("batch", "seq"), dtype="float32"),
                axes: R.Tensor((1,), dtype="int64"),
            ) -> R.Tensor(("batch",), dtype="float32"):
                batch = T.int64(is_size_var=True)
                seq = T.int64(is_size_var=True)
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((batch,), dtype="float32") = R.broadcast_to(
                        R.const(1, "float32"), R.shape([batch])
                    )
                    R.output(gv)
                return gv

        expected = ExpectedWithoutReshapeFlatten

    tvm.ir.assert_structural_equal(tvm_model["main"].without_attr("params"), expected["main"])


def test_multi_inputs_with_same_symbolic_shape():
    concat_node = helper.make_node("Concat", ["data1", "data2"], ["output"], axis=1)

    graph = helper.make_graph(
        [concat_node],
        "test_multi_symbolic_shape_input",
        inputs=[
            helper.make_tensor_value_info("data1", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("data2", TensorProto.FLOAT, ["batch", 1]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
    )
    model = helper.make_model(graph, producer_name="test_multi_symbolic_shape_input")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data1: R.Tensor(("batch", 1), dtype="float32"),
            data2: R.Tensor(("batch", 1), dtype="float32"),
        ) -> R.Tensor(("batch", 2), dtype="float32"):
            batch = T.int64(is_size_var=True)
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((batch, 2), dtype="float32") = R.concat((data1, data2), axis=1)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_multi_ops_with_same_params():
    reshape_node_1 = helper.make_node("Reshape", ["a", "x"], ["b"])
    reshape_node_2 = helper.make_node("Reshape", ["b", "x"], ["c"])

    a_shape = [16]
    output_shape = [1, 16]

    graph = helper.make_graph(
        [reshape_node_1, reshape_node_2],
        "test_multi_ops_with_same_params",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor("x", TensorProto.INT64, [2], output_shape),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="test_multi_ops_with_same_params")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((16,), dtype="float32"),
            v: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((1, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 16), dtype="float32") = R.reshape(a, R.shape([1, 16]))
                gv: R.Tensor((1, 16), dtype="float32") = R.reshape(lv, R.shape([1, 16]))
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_params_names_start_with_onnx():
    reshape_node = helper.make_node("Reshape", ["a", "onnx::x"], ["b"])

    a_shape = [16]
    output_shape = [1, 16]

    graph = helper.make_graph(
        [reshape_node],
        "test_params_names_start_with_onnx",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor("onnx::x", TensorProto.INT64, [2], output_shape),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="test_params_names_start_with_onnx")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((16,), dtype="float32"),
            v: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((1, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 16), dtype="float32") = R.reshape(a, R.shape([1, 16]))
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_add():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A + B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A + B"), dtype="float32")) -> R.Tensor(("A", "B", "A + B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A + B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_subtract():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A - B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A - B"), dtype="float32")) -> R.Tensor(("A", "B", "A - B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A - B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_mul():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A * B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A * B"), dtype="float32")) -> R.Tensor(("A", "B", "A * B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A * B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_div_1():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    # this will result in a floordiv despite not using // since the operands are always int
    x_shape = ["A", "B", "A / B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A // B"), dtype="float32")) -> R.Tensor(("A", "B", "A // B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A // B), dtype="float32") = x
                R.output(gv)
            return gv

    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_div_2():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A // B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A // B"), dtype="float32")) -> R.Tensor(("A", "B", "A // B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A // B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def _make_nms_expected(
    boxes_shape,
    scores_shape,
    center_point_box=0,
    nms_params=None,
):
    boxes = relax.Var("boxes", relax.TensorType(boxes_shape, "float32"))
    scores = relax.Var("scores", relax.TensorType(scores_shape, "float32"))
    params = [boxes, scores]
    for name, shape, dtype, _ in nms_params or []:
        params.append(relax.Var(name, relax.TensorType(shape, dtype)))

    def param_value(name, default, dtype):
        for param_name, _, _, value in nms_params or []:
            if param_name == name:
                default = value
                break
        if dtype == "float32":
            return np.float32(default).item()
        return int(default)

    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            nms_boxes = boxes
            if center_point_box != 0:
                split_result = bb.emit(relax.op.split(boxes, 4, axis=2))
                xc = bb.emit(split_result[0])
                yc = bb.emit(split_result[1])
                w = bb.emit(split_result[2])
                h = bb.emit(split_result[3])
                half_w = bb.emit(relax.op.divide(w, relax.const(2.0, "float32")))
                half_h = bb.emit(relax.op.divide(h, relax.const(2.0, "float32")))
                x1 = bb.emit(relax.op.subtract(xc, half_w))
                x2 = bb.emit(relax.op.add(xc, half_w))
                y1 = bb.emit(relax.op.subtract(yc, half_h))
                y2 = bb.emit(relax.op.add(yc, half_h))
                nms_boxes = bb.emit(relax.op.concat([y1, x1, y2, x2], axis=2))

            nms_out = bb.emit(
                relax.op.vision.all_class_non_max_suppression(
                    nms_boxes,
                    scores,
                    relax.const(param_value("max_output_boxes_per_class", 0, "int64"), "int64"),
                    relax.const(param_value("iou_threshold", 0.5, "float32"), "float32"),
                    relax.const(param_value("score_threshold", 0.0, "float32"), "float32"),
                    output_format="onnx",
                )
            )
            selected_indices = bb.emit(nms_out[0])
            gv = bb.emit_output(selected_indices)
        bb.emit_func_output(gv)

    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)
    return expected


def _assert_nms_import(
    model,
    boxes_shape,
    scores_shape,
    center_point_box=0,
    nms_params=None,
):
    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
    if nms_params:
        assert len(tvm_model["main"].attrs["params"]) == len(nms_params)
        tvm_model["main"] = tvm_model["main"].without_attr("params")

    tvm.ir.assert_structural_equal(
        tvm_model,
        _make_nms_expected(
            boxes_shape,
            scores_shape,
            center_point_box=center_point_box,
            nms_params=nms_params,
        ),
    )


def test_nms():
    """NonMaxSuppression should import as all_class_non_max_suppression."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 5, 4]  # batch_size, num_boxes, 4
    scores_shape = [1, 2, 5]  # batch_size, num_classes, num_boxes

    graph = helper.make_graph(
        [nms_node],
        "nms_test",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test")
    model.ir_version = 8
    model.opset_import[0].version = 11

    _assert_nms_import(
        model,
        boxes_shape,
        scores_shape,
        nms_params=[
            ("max_output_boxes_per_class", [1], "int64", 3),
            ("iou_threshold", [1], "float32", 0.5),
            ("score_threshold", [1], "float32", 0.1),
        ],
    )


def test_nms_scalar_shape1_constants():
    """Scalar params given as 1-D single-element constants must import (NumPy 2.x cast)."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
    )
    graph = helper.make_graph(
        [nms_node],
        "nms_scalar_shape1",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 5]),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.0]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    # Default import folds initializers to relax.Constant, exercising the scalar-cast path.
    from_onnx(model)


@pytest.mark.parametrize("with_explicit_max", [False, True])
def test_nms_max_output_boxes_per_class_zero(with_explicit_max: bool):
    """ONNX default for max_output_boxes_per_class should import as 0."""
    node_inputs = ["boxes", "scores"]
    initializer = []
    nms_params = None
    if with_explicit_max:
        node_inputs.append("max_output_boxes_per_class")
        initializer.append(
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [0])
        )
        nms_params = [("max_output_boxes_per_class", [1], "int64", 0)]

    nms_node = helper.make_node(
        "NonMaxSuppression",
        node_inputs,
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 4, 4]
    scores_shape = [1, 1, 4]
    graph = helper.make_graph(
        [nms_node],
        "nms_max_output_boxes_per_class_zero",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_max_output_boxes_per_class_zero")
    model.ir_version = 8
    model.opset_import[0].version = 11

    _assert_nms_import(
        model,
        boxes_shape,
        scores_shape,
        nms_params=nms_params,
    )


def test_nms_algorithm_correctness():
    """NMS import should pass max boxes, IoU, and score threshold constants."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 3, 4]  # batch_size, num_boxes, 4
    scores_shape = [1, 2, 3]  # batch_size, num_classes, num_boxes

    graph = helper.make_graph(
        [nms_node],
        "nms_test_correctness",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor(
                "max_output_boxes_per_class", TensorProto.INT64, [1], [2]
            ),  # Only 2 boxes per class
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),  # IoU threshold 0.5
            helper.make_tensor(
                "score_threshold", TensorProto.FLOAT, [1], [0.1]
            ),  # Score threshold 0.1
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [4, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_correctness")

    _assert_nms_import(
        model,
        boxes_shape,
        scores_shape,
        nms_params=[
            ("max_output_boxes_per_class", [1], "int64", 2),
            ("iou_threshold", [1], "float32", 0.5),
            ("score_threshold", [1], "float32", 0.1),
        ],
    )


def test_nms_iou_suppression():
    """NMS import should pass the IoU threshold constant."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 3, 4]
    scores_shape = [1, 1, 3]

    graph = helper.make_graph(
        [nms_node],
        "nms_test_iou_suppression",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [2]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),  # IoU threshold 0.5
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [2, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_iou_suppression")
    model.ir_version = 8
    model.opset_import[0].version = 11

    _assert_nms_import(
        model,
        boxes_shape,
        scores_shape,
        nms_params=[
            ("max_output_boxes_per_class", [1], "int64", 2),
            ("iou_threshold", [1], "float32", 0.5),
            ("score_threshold", [1], "float32", 0.1),
        ],
    )


def test_nms_max_boxes_limit():
    """NMS import should pass max_output_boxes_per_class."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 4, 4]
    scores_shape = [1, 1, 4]

    graph = helper.make_graph(
        [nms_node],
        "nms_test_max_boxes_limit",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor(
                "max_output_boxes_per_class", TensorProto.INT64, [1], [2]
            ),  # Limit to 2 boxes
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.1]),  # Low IoU threshold
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [2, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_max_boxes_limit")
    model.ir_version = 8
    model.opset_import[0].version = 11

    _assert_nms_import(
        model,
        boxes_shape,
        scores_shape,
        nms_params=[
            ("max_output_boxes_per_class", [1], "int64", 2),
            ("iou_threshold", [1], "float32", 0.1),
            ("score_threshold", [1], "float32", 0.1),
        ],
    )


def test_nms_score_threshold():
    """NMS import should pass the score threshold constant."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 3, 4]
    scores_shape = [1, 1, 3]

    graph = helper.make_graph(
        [nms_node],
        "nms_test_score_threshold",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.05]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [3, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_score_threshold")
    model.ir_version = 8
    model.opset_import[0].version = 11

    _assert_nms_import(
        model,
        boxes_shape,
        scores_shape,
        nms_params=[
            ("max_output_boxes_per_class", [1], "int64", 3),
            ("iou_threshold", [1], "float32", 0.1),
            ("score_threshold", [1], "float32", 0.05),
        ],
    )


# align_corners=None omits the attribute, exercising the ONNX default of 0.
@pytest.mark.parametrize("align_corners", [None, 0, 1])
def test_affine_grid(align_corners):
    attrs = {} if align_corners is None else {"align_corners": align_corners}
    affine_grid_node = helper.make_node("AffineGrid", ["theta", "size"], ["grid"], **attrs)

    graph = helper.make_graph(
        [affine_grid_node],
        "affine_grid_test",
        inputs=[
            helper.make_tensor_value_info("theta", TensorProto.FLOAT, [2, 2, 3]),
        ],
        initializer=[
            helper.make_tensor("size", TensorProto.INT64, [4], [2, 3, 16, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, [2, 16, 16, 2]),
        ],
    )

    model = helper.make_model(
        graph, producer_name="affine_grid_test", opset_imports=[helper.make_opsetid("", 20)]
    )
    tvm_model = from_onnx(model, opset=20, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    if align_corners:

        @I.ir_module
        class ExpectedAlignCorners:
            @R.function
            def main(
                theta: R.Tensor((2, 2, 3), dtype="float32"),
                size: R.Tensor((4,), dtype="int64"),
            ) -> R.Tensor((2, 16, 16, 2), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((2, 2, 16, 16), dtype="float32") = R.image.affine_grid(
                        theta, size=(16, 16), align_corners=True
                    )
                    lv1: R.Tensor((2, 16, 16, 2), dtype="float32") = R.permute_dims(
                        lv, axes=[0, 2, 3, 1]
                    )
                    gv: R.Tensor((2, 16, 16, 2), dtype="float32") = lv1
                    R.output(gv)
                return gv

        expected = ExpectedAlignCorners

    else:

        @I.ir_module
        class ExpectedDefaultAlignCorners:
            @R.function
            def main(
                theta: R.Tensor((2, 2, 3), dtype="float32"),
                size: R.Tensor((4,), dtype="int64"),
            ) -> R.Tensor((2, 16, 16, 2), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((2, 2, 16, 16), dtype="float32") = R.image.affine_grid(
                        theta, size=(16, 16), align_corners=False
                    )
                    lv1: R.Tensor((2, 16, 16, 2), dtype="float32") = R.permute_dims(
                        lv, axes=[0, 2, 3, 1]
                    )
                    gv: R.Tensor((2, 16, 16, 2), dtype="float32") = lv1
                    R.output(gv)
                return gv

        expected = ExpectedDefaultAlignCorners

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample(mode, padding_mode, align_corners):
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(
        graph, producer_name="grid_sample_test", opset_imports=[helper.make_opsetid("", 16)]
    )
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    X = relax.Var("X", relax.TensorType(x_shape, "float32"))
    grid = relax.Var("grid", relax.TensorType(grid_shape, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [X, grid]):
        with bb.dataflow():
            relax_grid = bb.emit(relax.op.permute_dims(grid, axes=[0, 3, 1, 2]))
            out = bb.emit_output(
                relax.op.image.grid_sample(
                    X,
                    relax_grid,
                    method=mode,
                    layout="NCHW",
                    padding_mode=padding_mode,
                    align_corners=bool(align_corners),
                )
            )
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample_5d(mode, padding_mode, align_corners):
    x_shape = [1, 1, 4, 4, 4]
    grid_shape = [1, 4, 4, 4, 3]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_5d_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(
        graph, producer_name="grid_sample_5d_test", opset_imports=[helper.make_opsetid("", 16)]
    )
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    X = relax.Var("X", relax.TensorType(x_shape, "float32"))
    grid = relax.Var("grid", relax.TensorType(grid_shape, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [X, grid]):
        with bb.dataflow():
            relax_grid = bb.emit(relax.op.permute_dims(grid, axes=[0, 4, 1, 2, 3]))
            out = bb.emit_output(
                relax.op.image.grid_sample(
                    X,
                    relax_grid,
                    method=mode,
                    layout="NCDHW",
                    padding_mode=padding_mode,
                    align_corners=bool(align_corners),
                )
            )
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)

    tvm.ir.assert_structural_equal(tvm_model, expected)


def test_grid_sample_5d_cubic_unsupported():
    x_shape = [1, 1, 4, 4, 4]
    grid_shape = [1, 2, 3, 5, 3]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="cubic",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_5d_cubic_unsupported_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_5d_cubic_unsupported_test")
    with pytest.raises(
        NotImplementedError,
        match="5D .*GridSample with mode='cubic' is not supported",
    ):
        from_onnx(model, opset=16, keep_params_in_input=True)


def test_grid_sample_4d_non_square_output_shape():
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 3, 5, 2]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="bilinear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_4d_non_square_output_shape_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_4d_non_square_output_shape_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            X: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 3, 5, 2), dtype="float32"),
        ) -> R.Tensor((1, 3, 3, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 5), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor((1, 3, 3, 5), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method="bilinear",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=False,
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_grid_sample_unsupported_rank():
    x_shape = [1, 3, 4]
    grid_shape = [1, 4, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="bilinear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_unsupported_rank_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, x_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_unsupported_rank_test")
    with pytest.raises(NotImplementedError, match="GridSample only supports 4D or 5D input"):
        from_onnx(model, opset=16, keep_params_in_input=True)


def test_grid_sample_linear_mode_translation():
    """Test that ONNX mode='linear' is correctly translated to 'bilinear'.

    The ONNX spec defines 'linear' as a valid mode for GridSample, but
    onnxruntime rejects it in practice. Real ONNX models exported from
    frameworks like PyTorch may still use 'linear'. We verify the translation
    by inspecting the Relax IR directly rather than running check_correctness.
    """
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="linear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_linear_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_linear_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            X: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 2, 2, 2), dtype="float32"),
        ) -> R.Tensor((1, 3, 2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor((1, 3, 2, 2), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method="bilinear",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=False,
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_grid_sample_cubic_mode_translation():
    """Test that ONNX mode='cubic' is correctly translated to 'bicubic'.

    The ONNX spec defines 'cubic' as a valid mode for GridSample, but
    TVM uses 'bicubic'. We verify the translation by inspecting the
    Relax IR directly rather than running check_correctness.
    """
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="cubic",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_cubic_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_cubic_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            X: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 2, 2, 2), dtype="float32"),
        ) -> R.Tensor((1, 3, 2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor((1, 3, 2, 2), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method="bicubic",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=False,
                )
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize(
    ("coordinate_transformation_mode", "rois"),
    [
        (
            "output_half_pixel",
            np.array([[1.0, 1.0, 6.0, 6.0], [2.0, 0.5, 7.0, 7.0]], dtype="float32"),
        ),
        ("half_pixel", np.array([[1.0, 1.0, 1.2, 1.2], [2.0, 0.5, 1.1, 1.1]], dtype="float32")),
    ],
)
def test_roi_align(coordinate_transformation_mode, rois):
    x_shape = [1, 4, 8, 8]
    rois_shape = [2, 4]
    batch_indices_shape = [2]
    out_shape = [2, 4, 3, 3]

    node = helper.make_node(
        "RoiAlign",
        inputs=["X", "rois", "batch_indices"],
        outputs=["Y"],
        output_height=3,
        output_width=3,
        sampling_ratio=2,
        spatial_scale=1.0,
        mode="avg",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )

    graph = helper.make_graph(
        [node],
        "roi_align_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, rois_shape),
            helper.make_tensor_value_info("batch_indices", TensorProto.INT64, batch_indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="roi_align_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    if coordinate_transformation_mode == "half_pixel":

        @I.ir_module
        class ExpectedRoiAlignHalfPixel:
            @R.function
            def main(
                X: R.Tensor((1, 4, 8, 8), dtype="float32"),
                rois: R.Tensor((2, 4), dtype="float32"),
                batch_indices: R.Tensor((2,), dtype="int64"),
            ) -> R.Tensor((2, 4, 3, 3), dtype="float32"):
                R.func_attr({"num_input": 3})
                with R.dataflow():
                    lv: R.Tensor((2, 1), dtype="int64") = R.expand_dims(batch_indices, axis=1)
                    lv1: R.Tensor((2, 1), dtype="float32") = R.astype(lv, dtype="float32")
                    lv2: R.Tensor((2, 4), dtype="float32") = R.add(
                        rois, R.const([-0.5, -0.5, -0.5, -0.5], "float32")
                    )
                    lv3: R.Tensor((2, 5), dtype="float32") = R.concat((lv1, lv2), axis=1)
                    gv: R.Tensor((2, 4, 3, 3), dtype="float32") = R.vision.roi_align(
                        X,
                        lv3,
                        pooled_size=(3, 3),
                        spatial_scale=1.0,
                        sample_ratio=2,
                        aligned=True,
                        layout="NCHW",
                        mode="avg",
                    )
                    R.output(gv)
                return gv

        expected = ExpectedRoiAlignHalfPixel

    else:

        @I.ir_module
        class ExpectedRoiAlignOutputHalfPixel:
            @R.function
            def main(
                X: R.Tensor((1, 4, 8, 8), dtype="float32"),
                rois: R.Tensor((2, 4), dtype="float32"),
                batch_indices: R.Tensor((2,), dtype="int64"),
            ) -> R.Tensor((2, 4, 3, 3), dtype="float32"):
                R.func_attr({"num_input": 3})
                with R.dataflow():
                    lv: R.Tensor((2, 1), dtype="int64") = R.expand_dims(batch_indices, axis=1)
                    lv1: R.Tensor((2, 1), dtype="float32") = R.astype(lv, dtype="float32")
                    lv2: R.Tensor((2, 5), dtype="float32") = R.concat((lv1, rois), axis=1)
                    gv: R.Tensor((2, 4, 3, 3), dtype="float32") = R.vision.roi_align(
                        X,
                        lv2,
                        pooled_size=(3, 3),
                        spatial_scale=1.0,
                        sample_ratio=2,
                        aligned=False,
                        layout="NCHW",
                        mode="avg",
                    )
                    R.output(gv)
                return gv

        expected = ExpectedRoiAlignOutputHalfPixel

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize(
    "cond_info, cond_true, cond_false",
    [
        (
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            np.array(True),
            np.array(False),
        ),
        (
            helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
            np.array([True]),
            np.array([False]),
        ),
    ],
    ids=["scalar_condition", "tensor_condition"],
)
def test_if(cond_info, cond_true, cond_false):
    """Test ONNX If operator with scalar and tensor bool conditions."""

    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    # then branch: x * 2.0
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    then_mul = helper.make_node("Mul", ["x", "two"], ["then_out"])
    then_out_info = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph([then_mul], "then_graph", [], [then_out_info], initializer=[two])

    # else branch: x * 3.0
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    else_mul = helper.make_node("Mul", ["x", "three"], ["else_out"])
    else_out_info = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul], "else_graph", [], [else_out_info], initializer=[three]
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["result"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    main_graph = helper.make_graph([if_node], "if_test", [cond_info, x_info], [result_info])
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if cond_true.shape == ():

        @I.ir_module
        class ExpectedScalarCondition:
            @R.function
            def main(
                cond: R.Tensor((), dtype="bool"),
                x: R.Tensor((3,), dtype="float32"),
            ) -> R.Tensor((3,), dtype="float32"):
                R.func_attr({"num_input": 2})
                if cond:
                    gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                    gv2: R.Tensor((3,), dtype="float32") = gv
                else:
                    gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                    gv2: R.Tensor((3,), dtype="float32") = gv1
                return gv2

        tvm.ir.assert_structural_equal(tvm_model, ExpectedScalarCondition)

    else:

        @I.ir_module
        class ExpectedTensorCondition:
            @R.function
            def main(
                cond: R.Tensor((1,), dtype="bool"),
                x: R.Tensor((3,), dtype="float32"),
            ) -> R.Tensor((3,), dtype="float32"):
                R.func_attr({"num_input": 2})
                if cond:
                    gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                    gv2: R.Tensor((3,), dtype="float32") = gv
                else:
                    gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                    gv2: R.Tensor((3,), dtype="float32") = gv1
                return gv2

        tvm.ir.assert_structural_equal(tvm_model, ExpectedTensorCondition)


def test_if_computed_condition():
    """Test If where condition is computed from another op in the main graph."""
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    reduce_node = helper.make_node(
        "ReduceSum", ["x"], ["x_sum"], keepdims=0, noop_with_empty_axes=0
    )
    greater_node = helper.make_node("Greater", ["x_sum", "zero"], ["cond"])

    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    then_mul = helper.make_node("Mul", ["x", "two"], ["then_out"])
    then_out_info = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph([then_mul], "then_graph", [], [then_out_info], initializer=[two])

    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    else_mul = helper.make_node("Mul", ["x", "three"], ["else_out"])
    else_out_info = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul], "else_graph", [], [else_out_info], initializer=[three]
    )

    if_node = helper.make_node(
        "If", inputs=["cond"], outputs=["result"], then_branch=then_graph, else_branch=else_graph
    )

    main_graph = helper.make_graph(
        [reduce_node, greater_node, if_node],
        "if_computed_cond",
        [x_info],
        [result_info],
        initializer=[zero],
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3,), dtype="float32"),
            zer: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 1})
            gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((), dtype="bool") = R.greater(gv, zer)
            if gv1:
                gv2: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv4: R.Tensor((3,), dtype="float32") = gv2
            else:
                gv3: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv4: R.Tensor((3,), dtype="float32") = gv3
            return gv4

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_if_multiple_outputs():
    """Test If operator where branches return multiple outputs."""
    cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    out1_info = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [3])
    out2_info = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [3])

    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])

    then_mul1 = helper.make_node("Mul", ["x", "two"], ["then_out1"])
    then_mul2 = helper.make_node("Mul", ["x", "three"], ["then_out2"])
    then_o1 = helper.make_tensor_value_info("then_out1", TensorProto.FLOAT, [3])
    then_o2 = helper.make_tensor_value_info("then_out2", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph(
        [then_mul1, then_mul2], "then_graph", [], [then_o1, then_o2], initializer=[two, three]
    )

    four = helper.make_tensor("four", TensorProto.FLOAT, [1], [4.0])
    five = helper.make_tensor("five", TensorProto.FLOAT, [1], [5.0])
    else_mul1 = helper.make_node("Mul", ["x", "four"], ["else_out1"])
    else_mul2 = helper.make_node("Mul", ["x", "five"], ["else_out2"])
    else_o1 = helper.make_tensor_value_info("else_out1", TensorProto.FLOAT, [3])
    else_o2 = helper.make_tensor_value_info("else_out2", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul1, else_mul2], "else_graph", [], [else_o1, else_o2], initializer=[four, five]
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out1", "out2"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    main_graph = helper.make_graph(
        [if_node], "if_multi_out", [cond_info, x_info], [out1_info, out2_info]
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"),
            x: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")):
            R.func_attr({"num_input": 2})
            if cond:
                gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv4: R.Tuple(
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = gv, gv1
            else:
                gv2: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([4.0], "float32"))
                gv3: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([5.0], "float32"))
                gv4: R.Tuple(
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = gv2, gv3
            gv5: R.Tensor((3,), dtype="float32") = gv4[0]
            gv6: R.Tensor((3,), dtype="float32") = gv4[1]
            return (gv5, gv6)

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_if_nested():
    """Test nested If operator inside a branch."""
    cond1_info = helper.make_tensor_value_info("cond1", TensorProto.BOOL, [])
    cond2_info = helper.make_tensor_value_info("cond2", TensorProto.BOOL, [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    # Inner then: x * 2
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    inner_then_mul = helper.make_node("Mul", ["x", "two"], ["inner_then_out"])
    inner_then_out_info = helper.make_tensor_value_info("inner_then_out", TensorProto.FLOAT, [3])
    inner_then_graph = helper.make_graph(
        [inner_then_mul], "inner_then", [], [inner_then_out_info], initializer=[two]
    )

    # Inner else: x * 3
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    inner_else_mul = helper.make_node("Mul", ["x", "three"], ["inner_else_out"])
    inner_else_out_info = helper.make_tensor_value_info("inner_else_out", TensorProto.FLOAT, [3])
    inner_else_graph = helper.make_graph(
        [inner_else_mul], "inner_else", [], [inner_else_out_info], initializer=[three]
    )

    # Outer then: nested If(cond2, x*2, x*3)
    inner_if = helper.make_node(
        "If",
        inputs=["cond2"],
        outputs=["outer_then_out"],
        then_branch=inner_then_graph,
        else_branch=inner_else_graph,
    )
    outer_then_out_info = helper.make_tensor_value_info("outer_then_out", TensorProto.FLOAT, [3])
    outer_then_graph = helper.make_graph([inner_if], "outer_then", [], [outer_then_out_info])

    # Outer else: x * 4
    four = helper.make_tensor("four", TensorProto.FLOAT, [1], [4.0])
    outer_else_mul = helper.make_node("Mul", ["x", "four"], ["outer_else_out"])
    outer_else_out_info = helper.make_tensor_value_info("outer_else_out", TensorProto.FLOAT, [3])
    outer_else_graph = helper.make_graph(
        [outer_else_mul], "outer_else", [], [outer_else_out_info], initializer=[four]
    )

    outer_if = helper.make_node(
        "If",
        inputs=["cond1"],
        outputs=["result"],
        then_branch=outer_then_graph,
        else_branch=outer_else_graph,
    )
    main_graph = helper.make_graph(
        [outer_if], "nested_if", [cond1_info, cond2_info, x_info], [result_info]
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            cond1: R.Tensor((), dtype="bool"),
            cond2: R.Tensor((), dtype="bool"),
            x: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 3})
            if cond2:
                gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv
            else:
                gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv1
            if cond1:
                gv4: R.Tensor((3,), dtype="float32") = gv2
            else:
                gv3: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([4.0], "float32"))
                gv4: R.Tensor((3,), dtype="float32") = gv3
            return gv4

    tvm.ir.assert_structural_equal(tvm_model, Expected)


# Helper that builds the ONNX graph for MatMulInteger so the tests don't repeat boilerplate code every time
def _make_matmulinteger_model(A_shape, B_shape, A_dtype, B_dtype, a_zp_array=None, b_zp_array=None):
    """Build a minimal single-node ONNX graph for MatMulInteger."""

    def np_dtype_to_onnx(dt):
        return {np.int8: TensorProto.INT8, np.uint8: TensorProto.UINT8}[dt]

    A_info = helper.make_tensor_value_info("A", np_dtype_to_onnx(A_dtype), A_shape)
    B_info = helper.make_tensor_value_info("B", np_dtype_to_onnx(B_dtype), B_shape)
    graph_inputs = [A_info, B_info]
    node_inputs = ["A", "B"]
    initializers = []

    def _add_zp(name, arr, dtype):
        onnx_dtype = np_dtype_to_onnx(dtype)
        shape = list(arr.shape)
        initializers.append(helper.make_tensor(name, onnx_dtype, shape, arr.flatten().tolist()))
        node_inputs.append(name)

    if a_zp_array is not None:
        _add_zp("a_zero_point", a_zp_array, A_dtype)
    elif b_zp_array is not None:
        node_inputs.append("")  # placeholder only needed if b_zp is present

    if b_zp_array is not None:
        _add_zp("b_zero_point", b_zp_array, B_dtype)

    out_info = helper.make_tensor_value_info("output", TensorProto.INT32, None)
    node = helper.make_node("MatMulInteger", inputs=node_inputs, outputs=["output"])
    graph = helper.make_graph(
        [node], "matmulinteger", graph_inputs, [out_info], initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 8
    return model


def _make_matmulinteger_expected(
    A_shape,
    B_shape,
    A_dtype,
    B_dtype,
    a_zp_shape=None,
    b_zp_shape=None,
):
    """Build expected Relax IR for MatMulInteger frontend lowering."""

    def _dtype(dtype):
        return np.dtype(dtype).name

    A = relax.Var("A", relax.TensorType(A_shape, _dtype(A_dtype)))
    B = relax.Var("B", relax.TensorType(B_shape, _dtype(B_dtype)))
    params = [A, B]

    bb = relax.BlockBuilder()
    if a_zp_shape is not None:
        a_zero_point = relax.Var("a_zero_point", relax.TensorType(a_zp_shape, _dtype(A_dtype)))
        params.append(a_zero_point)
    else:
        a_zero_point = None

    if b_zp_shape is not None:
        b_zero_point = relax.Var("b_zero_point", relax.TensorType(b_zp_shape, _dtype(B_dtype)))
        params.append(b_zero_point)
    else:
        b_zero_point = None

    with bb.function("main", params):
        with bb.dataflow():
            a = bb.emit(relax.op.astype(A, "int32"))
            if a_zero_point is not None:
                a_zp = bb.emit(relax.op.astype(a_zero_point, "int32"))
                if len(a_zp_shape) == 1:
                    a_zp = bb.emit(relax.op.expand_dims(a_zp, axis=-1))
                a = bb.emit(relax.op.subtract(a, a_zp))

            b = bb.emit(relax.op.astype(B, "int32"))
            if b_zero_point is not None:
                b_zp = bb.emit(relax.op.astype(b_zero_point, "int32"))
                if len(b_zp_shape) == 1:
                    b_zp = bb.emit(relax.op.expand_dims(b_zp, axis=0))
                b = bb.emit(relax.op.subtract(b, b_zp))

            out = bb.emit_output(relax.op.matmul(a, b, out_dtype="int32"))
        bb.emit_func_output(out)

    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 2)
    return expected


@pytest.mark.parametrize(
    "A_dtype,B_dtype,a_zp,b_zp",
    [
        (np.int8, np.int8, None, None),
        (np.uint8, np.uint8, None, None),
        (np.uint8, np.int8, None, None),
        (np.int8, np.uint8, None, None),
        (np.uint8, np.uint8, np.uint8(128), np.uint8(128)),
        (np.int8, np.int8, np.int8(1), np.int8(2)),
    ],
)
def test_matmulinteger(A_dtype, B_dtype, a_zp, b_zp):
    """2-D MatMulInteger should import dtype casts and zero-point subtraction."""
    model = _make_matmulinteger_model(
        [4, 8],
        [8, 6],
        A_dtype,
        B_dtype,
        a_zp_array=np.array(a_zp, dtype=A_dtype) if a_zp is not None else None,
        b_zp_array=np.array(b_zp, dtype=B_dtype) if b_zp is not None else None,
    )
    tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)
    if a_zp is not None:
        assert len(tvm_model["main"].attrs["params"]) == 2
        tvm_model["main"] = tvm_model["main"].without_attr("params")

    tvm.ir.assert_structural_equal(
        tvm_model,
        _make_matmulinteger_expected(
            [4, 8],
            [8, 6],
            A_dtype,
            B_dtype,
            a_zp_shape=[] if a_zp is not None else None,
            b_zp_shape=[] if b_zp is not None else None,
        ),
    )


@pytest.mark.parametrize(
    "A_shape,B_shape,a_zp,b_zp",
    [
        ((2, 4, 8), (2, 8, 6), np.int8(1), np.int8(2)),  # 3-D batched
        ((2, 3, 4, 8), (2, 3, 8, 6), np.int8(1), np.int8(2)),  # 4-D batched
    ],
)
def test_matmulinteger_batched(A_shape, B_shape, a_zp, b_zp):
    """Batched MatMulInteger should import as batched Relax matmul."""
    model = _make_matmulinteger_model(
        list(A_shape),
        list(B_shape),
        np.int8,
        np.int8,
        a_zp_array=np.array(a_zp, dtype=np.int8),
        b_zp_array=np.array(b_zp, dtype=np.int8),
    )
    tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 2
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    tvm.ir.assert_structural_equal(
        tvm_model,
        _make_matmulinteger_expected(
            list(A_shape),
            list(B_shape),
            np.int8,
            np.int8,
            a_zp_shape=[],
            b_zp_shape=[],
        ),
    )


def test_matmulinteger_per_channel_zp():
    """1-D zero points should expand for per-row/per-column MatMulInteger."""
    a_zp = np.arange(4, dtype=np.int8)  # shape [M=4], per-row
    b_zp = np.arange(6, dtype=np.int8)  # shape [N=6], per-col

    model = _make_matmulinteger_model(
        [4, 8], [8, 6], np.int8, np.int8, a_zp_array=a_zp, b_zp_array=b_zp
    )
    tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 2
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    tvm.ir.assert_structural_equal(
        tvm_model,
        _make_matmulinteger_expected(
            [4, 8],
            [8, 6],
            np.int8,
            np.int8,
            a_zp_shape=[4],
            b_zp_shape=[6],
        ),
    )


@pytest.mark.parametrize(
    ("pooled_shape", "rois"),
    [
        ((1, 1), np.array([[0.0, 1.0, 1.0, 6.0, 6.0], [0.0, 0.0, 0.0, 7.0, 7.0]], dtype="float32")),
        (
            (2, 3),
            np.array([[0.0, 1.2, 0.5, 6.8, 7.0], [0.0, -1.0, 2.0, 3.5, 5.2]], dtype="float32"),
        ),
        (
            (2, 2),
            np.array(
                [[0.0, 100.0, 100.0, 110.0, 110.0], [0.0, 1.0, 1.0, 6.0, 6.0]], dtype="float32"
            ),
        ),
    ],
)
def test_max_roi_pool(pooled_shape, rois):
    x_shape = [1, 4, 8, 8]
    out_shape = [2, 4, pooled_shape[0], pooled_shape[1]]

    node = helper.make_node(
        "MaxRoiPool",
        inputs=["X", "rois"],
        outputs=["Y"],
        pooled_shape=pooled_shape,
        spatial_scale=1.0,
    )

    graph = helper.make_graph(
        [node],
        "max_roi_pool_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, [2, 5]),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="max_roi_pool_test")
    inputs = {
        "X": rg.standard_normal(size=x_shape).astype("float32"),
        "rois": rois,
    }
    check_correctness(model, inputs=inputs, opset=16, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("op_name", ["ArgMax", "ArgMin"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("keepdims", [True, False])
def test_arg_min_max_select_last_index(op_name, axis, keepdims):
    """select_last_index=1 should lower to flip + argreduce + index remap."""
    shape = [3, 4, 5]

    node = helper.make_node(
        op_name,
        inputs=["data"],
        outputs=["out"],
        axis=axis,
        keepdims=int(keepdims),
        select_last_index=1,
    )

    out_shape = list(shape)
    if keepdims:
        out_shape[axis] = 1
    else:
        out_shape.pop(axis)

    graph = helper.make_graph(
        [node],
        "arg_select_last_index_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, out_shape)],
    )
    model = helper.make_model(graph, producer_name="arg_select_last_index_test")
    tvm_model = from_onnx(model, opset=12, keep_params_in_input=True)

    data = relax.Var("data", relax.TensorType(shape, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [data]):
        with bb.dataflow():
            flipped = bb.emit(relax.op.flip(data, axis=axis))
            if op_name == "ArgMax":
                reduced = bb.emit(relax.op.argmax(flipped, axis=axis, keepdims=keepdims))
            else:
                reduced = bb.emit(relax.op.argmin(flipped, axis=axis, keepdims=keepdims))
            out = bb.emit_output(relax.op.subtract(relax.const(shape[axis] - 1, "int64"), reduced))
        bb.emit_func_output(out)
    expected = bb.get()
    expected["main"] = expected["main"].with_attr("num_input", 1)

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("op_name", ["ArgMax", "ArgMin"])
def test_arg_min_max_select_last_index_no_tie(op_name):
    """select_last_index=0 should keep direct argreduce lowering."""
    shape = [4, 5]
    node = helper.make_node(
        op_name,
        inputs=["data"],
        outputs=["out"],
        axis=1,
        keepdims=1,
        select_last_index=0,
    )
    graph = helper.make_graph(
        [node],
        "arg_no_tie_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, [4, 1])],
    )
    model = helper.make_model(graph, producer_name="arg_no_tie_test")
    tvm_model = from_onnx(model, opset=12, keep_params_in_input=True)

    if op_name == "ArgMax":

        @I.ir_module
        class ExpectedArgMax:
            @R.function
            def main(
                data: R.Tensor((4, 5), dtype="float32"),
            ) -> R.Tensor((4, 1), dtype="int64"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((4, 1), dtype="int64") = R.argmax(data, axis=1, keepdims=True)
                    R.output(gv)
                return gv

        expected = ExpectedArgMax

    else:

        @I.ir_module
        class ExpectedArgMin:
            @R.function
            def main(
                data: R.Tensor((4, 5), dtype="float32"),
            ) -> R.Tensor((4, 1), dtype="int64"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tensor((4, 1), dtype="int64") = R.argmin(data, axis=1, keepdims=True)
                    R.output(gv)
                return gv

        expected = ExpectedArgMin

    tvm.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize("op_name", ["ArgMax", "ArgMin"])
def test_arg_min_max_select_last_index_ir(op_name):
    """select_last_index=1 must lower to flip + argmax/argmin + subtract in the Relax IR."""
    shape = [3, 4, 5]

    node = helper.make_node(
        op_name,
        inputs=["data"],
        outputs=["out"],
        axis=1,
        keepdims=1,
        select_last_index=1,
    )
    graph = helper.make_graph(
        [node],
        "arg_select_last_index_ir_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, [3, 1, 5])],
    )
    model = helper.make_model(graph, producer_name="arg_select_last_index_ir_test")
    tvm_model = from_onnx(model, opset=12, keep_params_in_input=True)

    if op_name == "ArgMax":

        @I.ir_module
        class ExpectedArgMax:
            @R.function
            def main(
                data: R.Tensor((3, 4, 5), dtype="float32"),
            ) -> R.Tensor((3, 1, 5), dtype="int64"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((3, 4, 5), dtype="float32") = R.flip(data, axis=1)
                    lv1: R.Tensor((3, 1, 5), dtype="int64") = R.argmax(lv, axis=1, keepdims=True)
                    gv: R.Tensor((3, 1, 5), dtype="int64") = R.subtract(R.const(3, "int64"), lv1)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedArgMax)

    else:

        @I.ir_module
        class ExpectedArgMin:
            @R.function
            def main(
                data: R.Tensor((3, 4, 5), dtype="float32"),
            ) -> R.Tensor((3, 1, 5), dtype="int64"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((3, 4, 5), dtype="float32") = R.flip(data, axis=1)
                    lv1: R.Tensor((3, 1, 5), dtype="int64") = R.argmin(lv, axis=1, keepdims=True)
                    gv: R.Tensor((3, 1, 5), dtype="int64") = R.subtract(R.const(3, "int64"), lv1)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedArgMin)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_split_to_sequence_keepdims_0(axis: int):
    """keepdims=0, no split input: each chunk of size 1 has the split axis squeezed out."""
    shape = [3, 4, 5]
    out_shape = [s for i, s in enumerate(shape) if i != axis]

    split_to_seq_node = helper.make_node(
        "SplitToSequence",
        ["data"],  # no split input — keepdims applies here
        ["output"],
        axis=axis,
        keepdims=0,
    )
    graph = helper.make_graph(
        [split_to_seq_node],
        f"test_split_to_sequence_keepdims_0_axis{axis}",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="test_split_to_sequence_keepdims_0")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if axis == 0:

        @I.ir_module
        class ExpectedKeepdims0Axis0:
            @R.function
            def main(
                data: R.Tensor((3, 4, 5), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((4, 5), dtype="float32"),
                R.Tensor((4, 5), dtype="float32"),
                R.Tensor((4, 5), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tuple(
                        R.Tensor((1, 4, 5), dtype="float32"),
                        R.Tensor((1, 4, 5), dtype="float32"),
                        R.Tensor((1, 4, 5), dtype="float32"),
                    ) = R.split(data, indices_or_sections=3, axis=0)
                    lv1: R.Tensor((1, 4, 5), dtype="float32") = lv[0]
                    lv2: R.Tensor((1, 4, 5), dtype="float32") = lv[1]
                    lv3: R.Tensor((1, 4, 5), dtype="float32") = lv[2]
                    lv4: R.Tensor((4, 5), dtype="float32") = R.squeeze(lv1, axis=[0])
                    lv5: R.Tensor((4, 5), dtype="float32") = R.squeeze(lv2, axis=[0])
                    lv6: R.Tensor((4, 5), dtype="float32") = R.squeeze(lv3, axis=[0])
                    gv: R.Tuple(
                        R.Tensor((4, 5), dtype="float32"),
                        R.Tensor((4, 5), dtype="float32"),
                        R.Tensor((4, 5), dtype="float32"),
                    ) = lv4, lv5, lv6
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedKeepdims0Axis0)

    elif axis == 1:

        @I.ir_module
        class ExpectedKeepdims0Axis1:
            @R.function
            def main(
                data: R.Tensor((3, 4, 5), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((3, 5), dtype="float32"),
                R.Tensor((3, 5), dtype="float32"),
                R.Tensor((3, 5), dtype="float32"),
                R.Tensor((3, 5), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tuple(
                        R.Tensor((3, 1, 5), dtype="float32"),
                        R.Tensor((3, 1, 5), dtype="float32"),
                        R.Tensor((3, 1, 5), dtype="float32"),
                        R.Tensor((3, 1, 5), dtype="float32"),
                    ) = R.split(data, indices_or_sections=4, axis=1)
                    lv1: R.Tensor((3, 1, 5), dtype="float32") = lv[0]
                    lv2: R.Tensor((3, 1, 5), dtype="float32") = lv[1]
                    lv3: R.Tensor((3, 1, 5), dtype="float32") = lv[2]
                    lv4: R.Tensor((3, 1, 5), dtype="float32") = lv[3]
                    lv5: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv1, axis=[1])
                    lv6: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv2, axis=[1])
                    lv7: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv3, axis=[1])
                    lv8: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv4, axis=[1])
                    gv: R.Tuple(
                        R.Tensor((3, 5), dtype="float32"),
                        R.Tensor((3, 5), dtype="float32"),
                        R.Tensor((3, 5), dtype="float32"),
                        R.Tensor((3, 5), dtype="float32"),
                    ) = lv5, lv6, lv7, lv8
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedKeepdims0Axis1)

    else:

        @I.ir_module
        class ExpectedKeepdims0Axis2:
            @R.function
            def main(
                data: R.Tensor((3, 4, 5), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((3, 4), dtype="float32"),
                R.Tensor((3, 4), dtype="float32"),
                R.Tensor((3, 4), dtype="float32"),
                R.Tensor((3, 4), dtype="float32"),
                R.Tensor((3, 4), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tuple(
                        R.Tensor((3, 4, 1), dtype="float32"),
                        R.Tensor((3, 4, 1), dtype="float32"),
                        R.Tensor((3, 4, 1), dtype="float32"),
                        R.Tensor((3, 4, 1), dtype="float32"),
                        R.Tensor((3, 4, 1), dtype="float32"),
                    ) = R.split(data, indices_or_sections=5, axis=2)
                    lv1: R.Tensor((3, 4, 1), dtype="float32") = lv[0]
                    lv2: R.Tensor((3, 4, 1), dtype="float32") = lv[1]
                    lv3: R.Tensor((3, 4, 1), dtype="float32") = lv[2]
                    lv4: R.Tensor((3, 4, 1), dtype="float32") = lv[3]
                    lv5: R.Tensor((3, 4, 1), dtype="float32") = lv[4]
                    lv6: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv1, axis=[2])
                    lv7: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv2, axis=[2])
                    lv8: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv3, axis=[2])
                    lv9: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv4, axis=[2])
                    lv10: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv5, axis=[2])
                    gv: R.Tuple(
                        R.Tensor((3, 4), dtype="float32"),
                        R.Tensor((3, 4), dtype="float32"),
                        R.Tensor((3, 4), dtype="float32"),
                        R.Tensor((3, 4), dtype="float32"),
                        R.Tensor((3, 4), dtype="float32"),
                    ) = lv6, lv7, lv8, lv9, lv10
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedKeepdims0Axis2)


def test_split_to_sequence_keepdims_ignored_when_split_provided():
    """Per spec: keepdims is ignored when split input is provided.
    TVM follows the spec — output keeps the split axis even with keepdims=0."""
    split_node = make_constant_node("split", TensorProto.INT64, (), [1])
    split_to_seq_node = helper.make_node(
        "SplitToSequence",
        ["data", "split"],
        ["output"],
        axis=0,
        keepdims=0,
    )
    graph = helper.make_graph(
        [split_node, split_to_seq_node],
        "test_split_to_sequence_keepdims_ignored",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 5])],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [1, 5])],
    )
    model = helper.make_model(
        graph,
        producer_name="test_split_to_sequence_keepdims_ignored",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 8
    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((4, 5), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((1, 5), dtype="float32"),
            R.Tensor((1, 5), dtype="float32"),
            R.Tensor((1, 5), dtype="float32"),
            R.Tensor((1, 5), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((1, 5), dtype="float32"),
                    R.Tensor((1, 5), dtype="float32"),
                    R.Tensor((1, 5), dtype="float32"),
                    R.Tensor((1, 5), dtype="float32"),
                ) = R.split(data, indices_or_sections=4, axis=0)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_split_to_sequence_uneven_last_chunk(axis: int):
    """Spec: last chunk may be smaller if dim is not divisible by scalar split."""
    shape = [5, 4] if axis == 0 else [3, 5]
    split_node = make_constant_node("split", TensorProto.INT64, (), [2])
    split_to_seq_node = helper.make_node(
        "SplitToSequence", ["data", "split"], ["output"], axis=axis, keepdims=1
    )
    graph = helper.make_graph(
        [split_node, split_to_seq_node],
        f"test_split_to_sequence_uneven_axis{axis}",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, producer_name="test_split_to_sequence_uneven")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    if axis == 0:

        @I.ir_module
        class ExpectedUnevenAxis0:
            @R.function
            def main(
                data: R.Tensor((5, 4), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((2, 4), dtype="float32"),
                R.Tensor((2, 4), dtype="float32"),
                R.Tensor((1, 4), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((2, 4), dtype="float32"),
                        R.Tensor((2, 4), dtype="float32"),
                        R.Tensor((1, 4), dtype="float32"),
                    ) = R.split(data, indices_or_sections=3, axis=0)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedUnevenAxis0)

    else:

        @I.ir_module
        class ExpectedUnevenAxis1:
            @R.function
            def main(
                data: R.Tensor((3, 5), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((3, 2), dtype="float32"),
                R.Tensor((3, 2), dtype="float32"),
                R.Tensor((3, 1), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    gv: R.Tuple(
                        R.Tensor((3, 2), dtype="float32"),
                        R.Tensor((3, 2), dtype="float32"),
                        R.Tensor((3, 1), dtype="float32"),
                    ) = R.split(data, indices_or_sections=3, axis=1)
                    R.output(gv)
                return gv

        tvm.ir.assert_structural_equal(tvm_model, ExpectedUnevenAxis1)


def test_quantizelinear_singleton_qparams_opset10():
    """QuantizeLinear must treat shape-[1] scale/zp as scalar in opset10."""
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_singleton_qparams_opset10",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 3, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [4, 3, 2, 2])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.03125]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [1], [127]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.standard_normal((4, 3, 2, 2)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_dequantizelinear_singleton_qparams_opset10():
    """DequantizeLinear must treat shape-[1] scale/zp as scalar in opset10."""
    node = helper.make_node("DequantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "dequantizelinear_singleton_qparams_opset10",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [64])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.125]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [1], [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.integers(low=0, high=255, size=(64,), dtype=np.uint8)
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_quantizelinear_optional_zero_point_opset13():
    """ONNX allows missing zero_point input; importer should default it to 0 (uint8)."""
    node = helper.make_node("QuantizeLinear", ["x", "scale"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_optional_zero_point_opset13",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 5])],
        initializer=[helper.make_tensor("scale", TensorProto.FLOAT, [], [0.2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    x = rg.standard_normal((2, 5)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=13, check_dtypes=True)


def test_dynamicquantizelinear_opset11():
    """DynamicQuantizeLinear should import as quantization helper ops."""
    node = helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "y_scale", "y_zero_point"])
    graph = helper.make_graph(
        [node],
        "dynamicquantizelinear_opset11",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3, 4]),
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, []),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 3, 4), dtype="uint8"),
            R.Tensor((), dtype="float32"),
            R.Tensor((), dtype="uint8"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.max(x, axis=None, keepdims=False)
                lv1: R.Tensor((), dtype="float32") = R.maximum(R.const(0.0, "float32"), lv)
                lv2: R.Tensor((), dtype="float32") = R.min(x, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.minimum(R.const(0.0, "float32"), lv2)
                lv4: R.Tensor((), dtype="float32") = R.subtract(lv1, lv3)
                lv5: R.Tensor((), dtype="float32") = R.divide(lv4, R.const(255.0, "float32"))
                lv6: R.Tensor((), dtype="float32") = R.divide(lv3, lv5)
                lv7: R.Tensor((), dtype="float32") = R.subtract(R.const(0.0, "float32"), lv6)
                lv8: R.Tensor((), dtype="float32") = R.clip(lv7, R.prim_value(0), R.prim_value(255))
                lv9: R.Tensor((), dtype="float32") = R.round(lv8)
                lv10: R.Tensor((), dtype="uint8") = R.astype(lv9, dtype="uint8")
                lv11: R.Tensor((2, 3, 4), dtype="uint8") = R.quantize(
                    x, lv5, lv10, out_dtype="uint8", axis=0
                )
                gv: R.Tuple(
                    R.Tensor((2, 3, 4), dtype="uint8"),
                    R.Tensor((), dtype="float32"),
                    R.Tensor((), dtype="uint8"),
                ) = (lv11, lv5, lv10)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_quantizelinear_default_axis_opset10():
    """opset10 QuantizeLinear should honor default axis=1 (not hardcode axis=0)."""
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_axis_opset10",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3, 4])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [3], [0.05, 0.1, 0.2]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [3], [1, 127, 250]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.standard_normal((2, 3, 4)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_dequantizelinear_default_axis_opset10():
    """opset10 DequantizeLinear should honor default axis=1 (not hardcode axis=0)."""
    node = helper.make_node("DequantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "dequantizelinear_axis_opset10",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [3], [0.05, 0.1, 0.2]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [3], [1, 127, 250]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.integers(low=0, high=255, size=(2, 3, 4), dtype=np.uint8)
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


if __name__ == "__main__":
    tvm.testing.main()
