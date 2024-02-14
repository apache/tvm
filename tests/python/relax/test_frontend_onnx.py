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
"""
ONNX testcases
================
This file is a test script to test Relax ONNX frontend coverage.
"""

from typing import Dict, Optional

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import ModelProto, TensorProto, helper, mapping

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.script import relax as R
from tvm.script import tir as T

bg = np.random.MT19937(0)
rg = np.random.Generator(bg)


def generate_random_inputs(
    model: ModelProto, inputs: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    input_values = {}
    # Iterate through model inputs and extract their shape.
    for i in model.graph.input:
        if inputs is not None and i.name in inputs and inputs[i.name] is not None:
            input_values[i.name] = inputs[i.name]
            continue
        shape = []
        for dim in i.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)

        # Extract datatype for the input.
        if i.type.tensor_type.elem_type:
            dtype = str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type])
        else:
            dtype = "float32"

        # Generate random inputs for each input.
        if dtype == "bool":
            # random_value = np.random.choice(a=[False, True], size=shape)
            random_value = rg.choice(a=[False, True], size=shape)
        else:
            # random_value = np.random.normal(size=shape).astype(dtype)
            random_value = rg.standard_normal(size=shape).astype(dtype)
        input_values[i.name] = random_value

    return input_values


def check_correctness(
    model: ModelProto,
    inputs: Optional[Dict[str, np.ndarray]] = None,
    ir_version: int = 8,
    opset: int = 14,
    atol: float = 1e-5,
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
        ex = relax.build(tvm_model, target="llvm")
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
    if isinstance(tvm_output, tvm.nd.NDArray):
        tvm_output = [tvm_output]
    # If the output is a shape tuple, convert it to an ndarray for comparison.
    if isinstance(tvm_output, tvm.runtime.ShapeTuple):
        tvm_output = [tvm.nd.array([int(i) for i in tvm_output])]

    tvm_num_outputs = len(tvm_output)
    # Shape tuples need to be handled specially.
    if isinstance(tvm_output, tvm.runtime.ShapeTuple):
        tvm_num_outputs = 1

    # Check that number of outputs match.

    assert tvm_num_outputs == len(ort_output), "Unequal number of outputs"

    for (tvm_out, ort_out) in zip(tvm_output, ort_output):
        # TODO Allow configurable tolerance.
        # Sometimes None is used to indicate an unused output.
        if ort_out is not None:
            tvm.testing.assert_allclose(tvm_out.numpy(), ort_out, atol=atol)


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


def verify_unary(op_name, shape, attrs={}, domain=None, dtype=TensorProto.FLOAT):
    test_node = helper.make_node(op_name, ["x"], ["y"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "elemwise_test",
        inputs=[
            helper.make_tensor_value_info("x", dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", dtype, shape)],
    )

    model = helper.make_model(graph, producer_name="elemwise_test")
    check_correctness(model)


def verify_binary(op_name, shape_a, shape_b, shape_c, attrs={}, domain=None):
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "binary_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, shape_a),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, shape_b),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, shape_c)],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    check_correctness(model)


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


def verify_ternary(op_name, shape_a, shape_b, shape_c, shape_d, attrs={}, domain=None):
    test_node = helper.make_node(op_name, ["a", "b", "c"], ["d"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "ternary_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, shape_a),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, shape_b),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, shape_c),
        ],
        outputs=[helper.make_tensor_value_info("d", TensorProto.FLOAT, shape_d)],
    )

    model = helper.make_model(graph, producer_name="ternary_test")
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


def test_concat():
    verify_binary("Concat", [1, 32], [1, 32], [2, 32], attrs={"axis": 0})


def test_add():
    verify_binary("Add", [1, 32], [1, 32], [1, 32])


def test_mul():
    verify_binary("Mul", [1, 32], [1, 32], [1, 32])


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


def test_gather():
    def _verify_gather(data_shape, indices, out_shape, axis=0):
        gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=axis)

        if isinstance(indices, (list, tuple)):
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

        model = helper.make_model(graph, producer_name="gather_test")
        input_values = {
            "data": np.random.randn(*data_shape).astype("float32"),
            "indices": np.array(indices).astype("int64"),
        }
        check_correctness(model, inputs=input_values)

    _verify_gather([5, 4, 3, 2], [0, 1, 3], [3, 4, 3, 2])
    _verify_gather([3], 0, [])
    _verify_gather([3, 3], [[0, 2]], [3, 1, 2], 1)


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
    check_correctness(model)


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
    input_values = {
        "data": np.random.randn(*in_shape).astype("float32"),
    }
    model = helper.make_model(graph, producer_name="reshape_test")
    check_correctness(model, inputs=input_values)


def test_div():
    verify_binary("Div", [32, 32], [32, 32], [32, 32])


def test_sigmoid():
    verify_unary("Sigmoid", [32, 32])


def test_softmax():
    verify_unary("Softmax", [32, 32, 32])


def test_transpose():
    verify_unary("Transpose", [32, 32, 32], attrs={"perm": [1, 2, 0]})


def test_unsqueeze():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [3], vals=[0, 2, 3])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_test")
    check_correctness(model)


def test_gelu():
    verify_unary("Gelu", [32, 32], domain="com.microsoft")


def test_bias_gelu():
    verify_binary("BiasGelu", [32, 32], [32], [32, 32], domain="com.microsoft")


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
    check_correctness(model)


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
    check_correctness(model)


def test_tanh():
    verify_unary("Tanh", [9, 8, 7, 6])


def test_sqrt():
    verify_unary("Sqrt", [32, 32])


def test_relu():
    verify_unary("Relu", [32, 32])


def test_tril():
    verify_unary("Trilu", [3, 5, 5], attrs={"upper": False})


def test_triu():
    verify_unary("Trilu", [3, 5, 5], attrs={"upper": True})


def test_elu():
    verify_unary("Elu", [32, 32])


def test_conv():
    def _verify_conv(input_shape, weight_shape, output_shape):
        bias_shape = [output_shape[1]]
        conv_node = helper.make_node("Conv", ["x", "w", "b"], ["y"])
        graph = helper.make_graph(
            [conv_node],
            "conv_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, bias_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="conv_test")
        check_correctness(model, atol=1e-4)

    # Conv1D
    _verify_conv([3, 12, 32], [4, 12, 3], [3, 4, 30])
    # Conv2D
    _verify_conv([3, 12, 32, 32], [4, 12, 3, 3], [3, 4, 30, 30])
    # Conv3D
    _verify_conv([3, 12, 32, 32, 32], [4, 12, 3, 3, 3], [3, 4, 30, 30, 30])


def test_pow():
    verify_binary("Pow", [32, 32], [32, 32], [32, 32])


def test_erf():
    verify_unary("Erf", [32, 32], dtype=TensorProto.FLOAT)
    verify_unary("Erf", [32, 32], dtype=TensorProto.FLOAT16)


@pytest.mark.parametrize("reverse", [False])
@pytest.mark.parametrize("exclusive", [False])
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

    model = helper.make_model(graph, producer_name="cumsum_test")
    check_correctness(model)


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

    model = helper.make_model(graph, producer_name="squeeze_test")
    check_correctness(model, opset=13)


def test_const():
    shape = [32, 32]
    const_node = helper.make_node(
        "Constant",
        [],
        ["y"],
        value=helper.make_tensor(
            "value", TensorProto.FLOAT, shape, np.random.rand(*shape).astype(np.float32).flatten()
        ),
    )
    graph = helper.make_graph(
        [const_node],
        "const_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="const_test")
    check_correctness(model)


def test_sub():
    verify_binary("Sub", [32, 16], [32, 16], [32, 16])


def test_min():
    verify_binary("Min", [32, 16], [32, 16], [32, 16])


def test_max():
    verify_binary("Max", [32, 16], [32, 16], [32, 16])


def test_sin():
    verify_unary("Sin", [32, 16])


def test_cos():
    verify_unary("Cos", [32, 16])


def test_identity():
    verify_unary("Identity", [32, 16])


def test_neg():
    verify_unary("Neg", [32, 16])


def test_abs():
    verify_unary("Abs", [32, 16])


def test_log():
    verify_unary("Log", [32, 16])


def test_exp():
    verify_unary("Exp", [32, 16])


def test_instance_norm():
    verify_ternary(
        "InstanceNormalization", [1, 3, 32, 32], [3], [3], [1, 3, 32, 32], attrs={"epsilon": 1e-12}
    )
    verify_ternary(
        "InstanceNormalization", [1, 32, 32], [32], [32], [1, 32, 32], attrs={"epsilon": 1e-12}
    )


def test_layer_norm():
    layer_norm_node = helper.make_node("LayerNormalization", ["a", "b", "c"], ["d"], epsilon=1e-12)

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("d", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)


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
        check_correctness(
            model,
            inputs={"input": input_, "skip": skip, "gamma": gamma, "beta": beta, "bias": bias},
        )

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

        inputs = {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "word_embedding": word_embedding,
            "position_embedding": position_embedding,
            "segment_embedding": segment_embedding,
            "gamma": gamma,
            "beta": beta,
        }
        check_correctness(model, inputs=inputs)

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


def create_reduce_test_parameters():
    output = []
    for value in [True, False]:
        output.append(("ReduceMax", value))
        output.append(("ReduceMean", value))
        output.append(("ReduceMin", value))
        output.append(("ReduceProd", value))
        output.append(("ReduceSum", value))
        output.append(("ReduceSumSquare", value))
        output.append(("ReduceLogSum", value))
        output.append(("ReduceLogSumExp", value))
        output.append(("ReduceL1", value))
        output.append(("ReduceL2", value))
    return output


@pytest.mark.parametrize("func, dynamic", create_reduce_test_parameters())
def test_all_reduce_funcs(func, dynamic):
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
        check_correctness(model, inputs_dict, opset=11, atol=1e-4)

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


@pytest.mark.parametrize("dynamic", [False, True])
# TODO(jwfromm) Current approach to dynamic expand is technically not well formed. Reenable once fixed.
@pytest.mark.skip("Produces ill-formed IR")
def test_expand(dynamic):
    if dynamic:
        # TODO: Support dynamic shape for Expand
        pytest.skip("Dynamic expand is not supported yet")

    def _test_expand(name, data, shape, ref_data):
        shape_array = np.array(shape)
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

        in_shape = list(data.shape)
        out_shape = list(ref_data.shape)
        if dynamic:
            in_shape = ["?" for _ in range(len(in_shape))]
            out_shape = ["?" for _ in range(len(out_shape))]
        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_teint64st",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, in_shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        check_correctness(model, inputs={"in": data})

    in_shape = (3, 1)
    shape = (3, 4)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = np.tile(data, 4)
    _test_expand("expand_with_dim_unchanged_test", data, shape, ref_data)


# TODO(jwfromm) Current approach to dynamic expand is technically not well formed. Reenable once fixed.
@pytest.mark.skip("Produces ill-formed IR")
def test_constantofshape():
    def verify_constantofshape(input_dim, value, dtype):
        fill_node = helper.make_node(
            "ConstantOfShape",
            ["input"],
            ["output"],
            value=helper.make_tensor(
                "value", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], (1,), (value,)
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
                    "output", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], input_dim
                )
            ],
        )

        model = helper.make_model(graph, producer_name="fill_test")
        input_np = np.array(input_dim).astype("int64")
        check_correctness(model, inputs={"input": input_np})

    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


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
        check_correctness(model)

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

        check_correctness(
            model,
            inputs={
                "input": input_,
                "weight": weight,
                "bias": bias,
                "mask_index": mask_index,
                "relative_position_bias": relative_position_bias,
            },
            # Maximum observed delta from 500 iterations was 2e-4.
            atol=1e-3,
        )
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


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad(dynamic):

    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, mode="constant", value=0.0):
        indata = np.random.normal(size=input_shape).astype(np.float32)
        #  numpy expect result
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        pads = np.array(pads)
        #  onnx graph
        if mode in ["edge", "reflect"]:
            outdata = np.pad(indata, pad_width=np_pads, mode=mode)
            node = helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"], mode=mode)
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))
                ],
                initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
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
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))
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
        check_correctness(model)

    verify_pad((2, 2), [0, 1, 0, 0], "constant", 0.0)
    verify_pad((2, 3), [1, 0, 0, 1], "constant", 0.0)
    verify_pad((3, 2), [0, 0, 1, 0], "constant", 5.0)
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect")


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
                "input", mapping.NP_TYPE_TO_TENSOR_TYPE[indata.dtype], indata_shape
            )
        ]

        if pass_split:
            if opset >= 13:
                np_split = np.array(split).astype(np.int64)
                initializer.append(
                    helper.make_tensor("split", TensorProto.INT64, list(np_split.shape), np_split)
                )
        node = helper.make_node(
            "Split",
            inputs=input_names,
            outputs=[f"output_{i}" for i in range(len(split_index))],
            axis=axis,
        )

        if pass_split and opset < 13:
            split_attr = helper.make_attribute("split", split)
            node.attribute.append(split_attr)

        graph = helper.make_graph(
            [node],
            "split_test",
            inputs=inputs,
            initializer=initializer,
            outputs=[
                helper.make_tensor_value_info(
                    f"output_{i}",
                    mapping.NP_TYPE_TO_TENSOR_TYPE[indata.dtype],
                    list(outdata_shapes[i]),
                )
                for i in range(len(split_index))
            ],
        )
        model = helper.make_model(graph, producer_name="split_test")
        check_correctness(model, inputs={"input": indata}, opset=opset)

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

        if dynamic:
            indata = np.random.normal(size=in_shape).astype(np.float32)
            in_shape = ["?" for _ in range(len(in_shape))]
            out_shape = ["?" for _ in range(len(out_shape))]

        graph = helper.make_graph(
            [node],
            "tile_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, in_shape),
            ],
            initializer=[
                helper.make_tensor("repeats", TensorProto.INT64, list(repeats.shape), repeats)
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name="tile_test")

        if dynamic:
            check_correctness(model, {"input": indata})
        else:
            check_correctness(model)

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z_array = np.tile(x, repeats)
    verify_tile(x.shape, repeats, z_array.shape)


def test_resize():
    resize_node = helper.make_node("Resize", ["X", "", "scales"], ["Y"], mode="cubic")

    graph = helper.make_graph(
        [resize_node],
        "resize_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
        initializer=[
            helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64]),
        ],
    )

    model = helper.make_model(graph, producer_name="resize_test")
    check_correctness(model)


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
    check_correctness(model)


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
    check_correctness(model)


def test_less():
    verify_compare("Less", [32, 32])


def test_less_equal():
    verify_compare("LessOrEqual", [32, 32])


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


def test_max_pool():
    # Pool2D
    verify_unary(
        "MaxPool",
        [1, 1, 32, 32],
        dict(
            auto_pad="NOTSET",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
    )
    # Pool2D with stride
    verify_unary(
        "MaxPool",
        [1, 1, 32, 32],
        dict(
            auto_pad="NOTSET",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        ),
    )
    # Pool2D with stride and autopadding
    verify_unary(
        "MaxPool",
        [1, 1, 32, 32],
        dict(
            auto_pad="SAME_UPPER",
            kernel_shape=[3, 7],
            pads=None,
            strides=[3, 2],
        ),
    )
    verify_unary(
        "MaxPool",
        [1, 1, 32, 32],
        dict(
            auto_pad="SAME_LOWER",
            kernel_shape=[3, 3],
            pads=None,
            strides=[2, 2],
        ),
    )
    verify_unary(
        "MaxPool",
        [1, 1, 32, 32],
        dict(
            auto_pad="VALID",
            kernel_shape=[3, 3],
            pads=None,
            strides=[2, 2],
        ),
    )
    verify_unary(
        "MaxPool",
        [1, 1, 32, 32],
        dict(
            auto_pad="SAME_UPPER",
            kernel_shape=[3, 3],
            pads=None,
        ),
    )


def test_global_average_pool():
    verify_unary("GlobalAveragePool", [1, 3, 32, 32])


def test_flatten():
    verify_unary("Flatten", [1, 3, 32, 32], attrs={"axis": 0})
    verify_unary("Flatten", [1, 3, 32, 32], attrs={"axis": -1})
    verify_unary("Flatten", [1, 3, 32, 32], attrs={"axis": 2})


def test_greater():
    verify_compare("Greater", [32, 32])
    verify_compare("Greater", [64, 16])


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


def test_reciprocal():
    verify_unary("Reciprocal", [3, 32, 32])


def test_symbolic_shape_deduction():
    index_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["indices"],
        value=helper.make_tensor("indices", TensorProto.INT64, [], [0]),
    )
    shape_node = helper.make_node("Shape", ["data"], ["shape_output"])
    gather_node = helper.make_node("Gather", ["shape_output", "indices"], ["gather_output"])
    unsqueeze_node = helper.make_node("Unsqueeze", ["gather_output", "axes"], ["unsqueeze_output"])
    constant_of_shape_node = helper.make_node(
        "ConstantOfShape",
        ["unsqueeze_output"],
        ["output"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [1]),
    )
    graph = helper.make_graph(
        [index_node, shape_node, gather_node, unsqueeze_node, constant_of_shape_node],
        "test_shape_deduction",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, ["batch", "seq"]),
        ],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [1], vals=[0])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [1])],
    )
    model = helper.make_model(graph, producer_name="test_shape_deduction")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @R.function
    def expected(
        data: R.Tensor(("batch", "seq"), dtype="float32")
    ) -> R.Tensor(dtype="float32", ndim=1):
        batch = T.int64()
        seq = T.int64()
        R.func_attr({"num_input": 1})
        with R.dataflow():
            gv: R.Tensor((batch,), dtype="float32") = R.broadcast_to(
                R.const(1, "float32"), R.shape([batch])
            )
            R.output(gv)
        return gv

    # TODO(siyuan): Enable assertion after fixing the SizeVar roundtrip issue
    # tvm.ir.assert_structural_equal(expected, tvm_model["main"])


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
    check_correctness(model)


if __name__ == "__main__":
    tvm.testing.main()
