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

from collections import OrderedDict
import sys

import numpy as np
import pytest

import tvm
from tvm import relay, TVMError
from tvm.ir.module import IRModule
from tvm.relay import testing, transform
from tvm.relay.testing import byoc
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.backend import Executor, Runtime
from tvm.micro import model_library_format as mlf
from aot_test_utils import (
    AOTTestModel,
    AOT_DEFAULT_RUNNER,
    generate_ref_data,
    convert_to_relay,
    compile_and_run,
    compile_models,
    parametrize_aot_options,
    create_relay_module_and_inputs_from_tflite_file,
)


def test_error_c_interface_with_packed_api():
    interface_api = "c"
    use_unpacked_api = False
    test_runner = AOT_DEFAULT_RUNNER

    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)

    with pytest.raises(tvm.TVMError, match="Packed interface required for packed operators"):
        compile_and_run(
            AOTTestModel(
                module=IRModule.from_expr(func), inputs={}, outputs=generate_ref_data(func, {})
            ),
            test_runner,
            interface_api,
            use_unpacked_api,
        )


@parametrize_aot_options
def test_conv_with_params(interface_api, use_unpacked_api, test_runner):
    RELAY_MODEL = """
#[version = "0.0.5"]
def @main(%data : Tensor[(1, 3, 64, 64), uint8], %weight : Tensor[(8, 3, 5, 5), int8]) {
    %1 = nn.conv2d(
         %data,
         %weight,
         padding=[2, 2],
         channels=8,
         kernel_size=[5, 5],
         data_layout="NCHW",
         kernel_layout="OIHW",
         out_dtype="int32");
  %1
}
"""
    mod = tvm.parser.fromtext(RELAY_MODEL)
    main_func = mod["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.ones(shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])

    params = {"weight": weight_data}
    inputs = {"data": input_data}
    output_list = generate_ref_data(mod, inputs, params)

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_add_with_params(interface_api, use_unpacked_api, test_runner):
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    func = relay.Function([x, y], z)

    x_in = np.ones((1, 10)).astype("float32")
    y_in = np.random.uniform(size=(1, 10)).astype("float32")

    params = {"x": x_in}
    inputs = {"y": y_in}
    output_list = generate_ref_data(func, inputs, params)

    compile_and_run(
        AOTTestModel(
            module=IRModule.from_expr(func),
            inputs=inputs,
            outputs=output_list,
            params=params,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
@pytest.mark.parametrize("groups,weight_shape", [(1, 32), (32, 1)])
def test_conv2d(interface_api, use_unpacked_api, test_runner, groups, weight_shape):
    """Test a subgraph with a single conv2d operator."""
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)

    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)

    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])

    output_list = generate_ref_data(mod, inputs)
    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_concatenate(interface_api, use_unpacked_api, test_runner):
    dtype = "float32"
    x = relay.var("x", shape=(10, 5), dtype=dtype)
    y = relay.var("y", shape=(10, 5), dtype=dtype)
    t = relay.var("z", shape=(), dtype=dtype)
    z = relay.concatenate((x, y), axis=1)
    z = relay.add(z, t)
    # Check result.
    func = relay.Function([x, y, t], z)
    x_data = np.random.rand(10, 5).astype(dtype)
    y_data = np.random.rand(10, 5).astype(dtype)
    t_data = np.random.uniform(size=()).astype(dtype)
    inputs = OrderedDict([("x", x_data), ("y", y_data), ("z", t_data)])

    output_list = generate_ref_data(func, inputs)
    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_nested_tuples(interface_api, use_unpacked_api, test_runner):
    x = relay.var("x", shape=(10,))
    x1 = x + relay.const(1.0)
    x2 = x1 + relay.const(1.0)
    x3 = x2 + relay.const(1.0)
    x4 = x3 + relay.const(1.0)
    out = relay.Tuple([x1, relay.Tuple([relay.Tuple([x2, x3]), x4])])
    func = relay.Function([x], out)

    x_data = np.random.uniform(size=(10,)).astype(np.float32)
    inputs = {"x": x_data}
    output_list = generate_ref_data(func, inputs)

    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_tuple_getitem(interface_api, use_unpacked_api, test_runner):
    func = relay.Function([], relay.TupleGetItem(relay.Tuple([relay.const(1), relay.const(2)]), 0))
    output_list = generate_ref_data(func, {})

    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs={}, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_id(interface_api, use_unpacked_api, test_runner):
    x = relay.var("x", "float32")
    ident = relay.Function([x], x)
    one = np.array(1.0, "float32")
    inputs = {"x": one}
    output_list = generate_ref_data(ident, inputs)

    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(ident), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_add_const(interface_api, use_unpacked_api, test_runner):
    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)
    output_list = generate_ref_data(func, {})

    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs={}, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_mul_param(interface_api, use_unpacked_api, test_runner):
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    func = relay.Function([x, y], relay.multiply(x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(1, 10).astype("float32")

    inputs = OrderedDict([("x", x_data), ("y", y_data)])
    output_list = generate_ref_data(func, inputs)

    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_subtract(interface_api, use_unpacked_api, test_runner):
    i = relay.var("i", shape=[], dtype="int32")
    sub = relay.subtract(i, relay.const(1, dtype="int32"))
    func = relay.Function([i], sub, ret_type=relay.TensorType([], "int32"))
    i_data = np.array(1, dtype="int32")
    inputs = {"i": i_data}
    output_list = generate_ref_data(func, inputs)
    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_tuple_output(interface_api, use_unpacked_api, test_runner):
    x = relay.var("x", shape=(6, 9))
    y = relay.split(x, 3).astuple()
    a = relay.TupleGetItem(y, 0)
    b = relay.TupleGetItem(y, 1)
    out = relay.Tuple([a, b])
    func = relay.Function([x], out)
    x_data = np.random.rand(6, 9).astype("float32")
    inputs = {"x": x_data}
    output_list = generate_ref_data(func, inputs)
    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@pytest.mark.parametrize(
    ["debug_calculated_workspaces", "workspace_byte_alignment"], [(True, 1), (True, 16), (False, 1)]
)
def test_mobilenet(debug_calculated_workspaces, workspace_byte_alignment):
    use_unpacked_api = True
    interface_api = "c"
    test_runner = AOT_DEFAULT_RUNNER

    # TODO(@Mousius) - Enable memory planning to take into account debug information
    debugging_memory_overhead = 1024 * 1024

    mod, params = testing.mobilenet.get_workload(batch_size=1)
    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    inputs = {"data": data}
    output_list = generate_ref_data(mod, inputs, params)
    compile_and_run(
        AOTTestModel(
            module=mod,
            inputs=inputs,
            outputs=output_list,
            params=params,
            extra_memory_in_bytes=debugging_memory_overhead,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
        workspace_byte_alignment=workspace_byte_alignment,
        debug_calculated_workspaces=debug_calculated_workspaces,
    )


@pytest.mark.parametrize("merge_compiler_regions", [False, True])
def test_byoc_microtvm(merge_compiler_regions):
    """This is a simple test to check BYOC capabilities of AOT - with and without merging compiler regions to test for https://github.com/apache/tvm/issues/9036"""
    use_unpacked_api = False
    interface_api = "packed"
    test_runner = AOT_DEFAULT_RUNNER

    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))

    # z0 = x + w0
    x_ = compiler_begin(x, "ccompiler")
    w0_ = compiler_begin(w0, "ccompiler")
    z0_ = relay.add(x_, w0_)
    z0 = compiler_end(z0_, "ccompiler")

    # z1 = z0 + w1
    z0__ = compiler_begin(z0, "ccompiler")
    w1_ = compiler_begin(w1, "ccompiler")
    z1_ = relay.add(z0__, w1_)
    z1 = compiler_end(z1_, "ccompiler")

    # z2 = z0 + z1
    z2 = relay.add(z0, z1)

    f = relay.Function([x, w0, w1], z2)
    mod = tvm.IRModule()
    mod["main"] = f

    if merge_compiler_regions:
        mod = transform.MergeCompilerRegions()(mod)

    mod = transform.PartitionGraph("mod_name")(mod)
    mod = transform.InferType()(mod)

    x_data = [("x", np.random.rand(10, 10).astype("float32"))]
    w_data = [("w{}".format(i), np.random.rand(10, 10).astype("float32")) for i in range(2)]

    map_inputs = OrderedDict(x_data + w_data)
    output_list = generate_ref_data(mod, map_inputs)
    compile_and_run(
        AOTTestModel(name="my_mod", module=mod, inputs=map_inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@pytest.mark.parametrize("merge_compiler_regions", [False, True])
def test_byoc_microtvm_multiple_subgraphs(merge_compiler_regions):
    """This is a test case to check BYOC capabilities of AOT with multiple sub graphs"""
    use_unpacked_api = False
    interface_api = "packed"
    test_runner = AOT_DEFAULT_RUNNER

    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))
    w2 = relay.var("w2", shape=(10, 10))
    w3 = relay.var("w3", shape=(10, 10))
    w4 = relay.var("w4", shape=(10, 10))
    w5 = relay.var("w5", shape=(10, 10))
    w6 = relay.var("w6", shape=(10, 10))
    w7 = relay.var("w7", shape=(10, 10))

    # C compiler
    z0 = relay.add(x, w0)
    p0 = relay.subtract(z0, w1)
    q0 = relay.multiply(p0, w2)

    z1 = relay.add(x, w3)
    p1 = relay.subtract(z1, w4)
    q1 = relay.multiply(p1, w5)

    # Other parts on TVM
    z2 = relay.add(x, w6)
    q2 = relay.subtract(z2, w7)

    r = relay.concatenate((q0, q1, q2), axis=0)
    f = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], r)
    mod = tvm.IRModule()
    ann = byoc.CcompilerAnnotator()
    mod["main"] = ann.visit(f)

    if merge_compiler_regions:
        mod = transform.MergeCompilerRegions()(mod)

    mod = tvm.relay.transform.PartitionGraph("mod_name")(mod)
    mod = tvm.relay.transform.InferType()(mod)

    x_data = np.random.rand(10, 10).astype("float32")
    w_data = []
    for _ in range(8):
        w_data.append(np.random.rand(10, 10).astype("float32"))

    map_inputs = OrderedDict([("x", x_data)] + [("w{}".format(i), w_data[i]) for i in range(8)])
    output_list = generate_ref_data(mod, map_inputs)
    input_list = [map_inputs["x"]]
    input_list.extend([map_inputs["w{}".format(i)] for i in range(8)])
    compile_and_run(
        AOTTestModel(name="my_mod", module=mod, inputs=map_inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_add_name_mangling_with_params(interface_api, use_unpacked_api, test_runner):
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    func = relay.Function([x, y], z)

    x_in = np.ones((1, 10)).astype("float32")
    y_in = np.random.uniform(size=(1, 10)).astype("float32")

    params = {"x": x_in}
    inputs = {"y": y_in}
    output_list = generate_ref_data(func, inputs, params)

    compile_and_run(
        AOTTestModel(
            name="my_mod",
            module=func,
            inputs=inputs,
            outputs=output_list,
            params=params,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_multiple_models(interface_api, use_unpacked_api, test_runner):
    # Identity model without params
    x = relay.var("x", "float32")
    mod1 = relay.Function([x], x)
    one = np.array(1.0, "float32")
    inputs1 = {"x": one}
    output_list1 = generate_ref_data(mod1, inputs1)
    params1 = None

    # Convolution model
    RELAY_MODEL = """
#[version = "0.0.5"]
def @main(%data : Tensor[(1, 3, 64, 64), uint8], %weight : Tensor[(8, 3, 5, 5), int8]) {
    %1 = nn.conv2d(
         %data,
         %weight,
         padding=[2, 2],
         channels=8,
         kernel_size=[5, 5],
         data_layout="NCHW",
         kernel_layout="OIHW",
         out_dtype="int32");
  %1
}
"""
    mod2 = tvm.parser.fromtext(RELAY_MODEL)
    main_func = mod2["main"]
    shape_dict = {p.name_hint: p.checked_type.concrete_shape for p in main_func.params}
    type_dict = {p.name_hint: p.checked_type.dtype for p in main_func.params}

    weight_data = np.ones(shape_dict["weight"]).astype(type_dict["weight"])
    input_data = np.ones(shape_dict["data"]).astype(type_dict["data"])

    params2 = {"weight": weight_data}
    inputs2 = {"data": input_data}
    output_list2 = generate_ref_data(mod2, inputs2, params2)

    compile_and_run(
        [
            AOTTestModel(
                name="mod1",
                module=mod1,
                inputs=inputs1,
                outputs=output_list1,
                params=params1,
            ),
            AOTTestModel(
                name="mod2",
                module=mod2,
                inputs=inputs2,
                outputs=output_list2,
                params=params2,
            ),
        ],
        test_runner,
        interface_api,
        use_unpacked_api,
    )


def test_quant_mobilenet_tfl():
    """Since in AOT we pass directly the output buffer from the user, in quantized networks sharing the output buffers is not possible.
    This is because the output data type is int8 and the intermediate buffer are int32 or int16. We use mobilenet quantized to stress this
    situation and verify that the output buffer sharing is disabled in AOT."""
    pytest.importorskip("tflite")

    import tvm.relay.testing.tf as tf_testing

    use_unpacked_api = True
    interface_api = "c"
    test_runner = AOT_DEFAULT_RUNNER

    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/"
        "models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        "mobilenet_v1_1.0_224_quant.tflite",
    )
    mod, inputs, params = create_relay_module_and_inputs_from_tflite_file(tflite_model_file)
    output_list = generate_ref_data(mod, inputs, params)
    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@parametrize_aot_options
def test_transpose(interface_api, use_unpacked_api, test_runner):
    """Test that non-inpleaceable operations (e.g., transpose) do not happen in-place."""

    dtype = "float32"
    x = relay.var("x", shape=(10, 5), dtype=dtype)
    y = relay.var("y", shape=(10, 5), dtype=dtype)
    t = relay.var("z", shape=(), dtype=dtype)
    a = relay.add(x, y)
    b = relay.transpose(a)
    z = relay.add(b, t)
    # Check result.
    func = relay.Function([x, y, t], z)
    x_data = np.random.rand(10, 5).astype(dtype)
    y_data = np.random.rand(10, 5).astype(dtype)
    t_data = np.random.uniform(size=()).astype(dtype)

    inputs = {"x": x_data, "y": y_data, "z": t_data}
    output_list = generate_ref_data(func, inputs)
    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
        enable_op_fusion=False,
    )


def test_name_sanitiser():
    """Test that input tensors with special characters in the name don't break compilation"""

    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_DEFAULT_RUNNER

    func = relay.var("input-x::2", "float32")
    ident = relay.Function([func], func)
    one = np.array(1.0, "float32")
    inputs = {"input-x::2": one}
    output_list = generate_ref_data(ident, inputs)

    compile_and_run(
        AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
        enable_op_fusion=False,
    )


def test_name_sanitiser_name_clash():
    """Test that 2 input tensors with names that clash once sanitized, generates an error"""

    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_DEFAULT_RUNNER

    dtype = "float32"
    x = relay.var("input::-1", shape=(10, 5), dtype=dtype)
    # Next 2 input tensor names will clash once sanitized.
    y = relay.var("input::-2", shape=(10, 5), dtype=dtype)
    t = relay.var("input:--2", shape=(), dtype=dtype)
    a = relay.add(x, y)
    b = relay.transpose(a)
    z = relay.add(b, t)
    # Check result.
    func = relay.Function([x, y, t], z)
    x_data = np.random.rand(10, 5).astype(dtype)
    y_data = np.random.rand(10, 5).astype(dtype)
    t_data = np.random.uniform(size=()).astype(dtype)

    inputs = {"input::-1": x_data, "input::-2": y_data, "input:--2": t_data}
    output_list = generate_ref_data(func, inputs)

    with pytest.raises(TVMError, match="Sanitized input tensor name clash"):
        compile_and_run(
            AOTTestModel(module=IRModule.from_expr(func), inputs=inputs, outputs=output_list),
            test_runner,
            interface_api,
            use_unpacked_api,
            enable_op_fusion=False,
        )


# This tests for deprecated AOT executor arguments
# TODO(Mousius) Remove deprecated arguments later
def test_deprecated_target_arguments(capsys):
    """Tests we can still use relay.build with -executor, -runtime and -link-params"""

    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_DEFAULT_RUNNER

    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    func = relay.Function([x, y], z)

    x_in = np.ones((1, 10)).astype("float32")
    y_in = np.random.uniform(size=(1, 10)).astype("float32")

    params = {"x": x_in}
    inputs = {"y": y_in}
    output_list = generate_ref_data(func, inputs, params)

    compile_and_run(
        AOTTestModel(
            module=IRModule.from_expr(func),
            inputs=inputs,
            outputs=output_list,
            params=params,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
        use_runtime_executor=False,
        target="c -executor=aot --link-params -runtime=c -interface-api=c --unpacked-api",
    )


def test_aot_codegen_backend_alloc_workspace_calls():
    """This test checks whether AoT lowering creates TVMBackendAllocWorkspace calls"""

    # The %data and %weight shapes in the following primitive Relay should create
    # small tensors that would get lowered to stack allocations in the CPU PrimFuncs.
    # However, the AoT executor codegen should retain them as TVMBAW calls
    relay_mod = tvm.parser.fromtext(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(1, 4, 4, 4), float32], %weight: Tensor[(4, 4, 3, 3), float32], src_layout="OIHW", dst_layout="OIHW4i4o") -> Tensor[(1, 4, 4, 4), float32] {
        %0 = fn (%p02: Tensor[(1, 4, 4, 4), float32], Primitive=1, hash="9332b3872fb5292c", src_layout="NCHW", dst_layout="NCHW4c") -> Tensor[(1, 1, 4, 4, 4), float32] {
            layout_transform(%p02, src_layout="NCHW", dst_layout="NCHW4c") /* ty=Tensor[(1, 1, 4, 4, 4), float32] */
        };
        %1 = fn (%p03: Tensor[(4, 4, 3, 3), float32], Primitive=1, hash="9f0b2b8a24a4dab3", src_layout="OIHW", dst_layout="OIHW4i4o") -> Tensor[(1, 1, 3, 3, 4, 4), float32] {
            layout_transform(%p03, src_layout="OIHW", dst_layout="OIHW4i4o") /* ty=Tensor[(1, 1, 3, 3, 4, 4), float32] */
        };
        %2 = %0(%data) /* ty=Tensor[(1, 1, 4, 4, 4), float32] */;
        %3 = %1(%weight) /* ty=Tensor[(1, 1, 3, 3, 4, 4), float32] */;
        %4 = fn (%p01: Tensor[(1, 1, 4, 4, 4), float32], %p1: Tensor[(1, 1, 3, 3, 4, 4), float32], out_layout="NCHW4c", kernel_layout="OIHW4i4o", Primitive=1, data_layout="NCHW4c") -> Tensor[(1, 1, 4, 4, 4), float32] {
                                                                                                                                                                                                                                                      nn.contrib_conv2d_NCHWc(%p01, %p1, padding=[1, 1, 1, 1], channels=4, kernel_size=[3, 3], data_layout="NCHW4c", kernel_layout="OIHW4i4o", out_layout="NCHW4c") /* ty=Tensor[(1, 1, 4, 4, 4), float32] */
        };
        %5 = %4(%2, %3) /* ty=Tensor[(1, 1, 4, 4, 4), float32] */;
        %6 = fn (%p0: Tensor[(1, 1, 4, 4, 4), float32], Primitive=1, src_layout="NCHW4c", dst_layout="NCHW") -> Tensor[(1, 4, 4, 4), float32] {
            layout_transform(%p0, src_layout="NCHW4c", dst_layout="NCHW") /* ty=Tensor[(1, 4, 4, 4), float32] */
        };
        %6(%5) /* ty=Tensor[(1, 4, 4, 4), float32] */
        }
        """
    )
    compiled_test_mods = compile_models(
        models=AOTTestModel(module=relay_mod, inputs=None, outputs=None),
        interface_api="c",
        use_unpacked_api=True,
    )
    source = compiled_test_mods[0].executor_factory.lib.imported_modules[0].get_source()
    # There should be three allocates created for three primitive relay function
    # calls in the main for the above relay snippet.
    assert source.count("TVMBackendAllocWorkspace") == 3


@pytest.mark.parametrize("constants_byte_alignment", [8, 16, 32])
def test_constants_alignment(constants_byte_alignment):
    """Test that constants_byte_alignment correctly sets constants byte alignment"""

    use_unpacked_api = True
    interface_api = "c"

    mod, params = testing.mobilenet.get_workload(batch_size=1)
    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    inputs = {"data": data}
    output_list = generate_ref_data(mod, inputs, params)
    target = f"c -constants-byte-alignment={constants_byte_alignment}"
    compiled_test_mods = compile_models(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        interface_api,
        use_unpacked_api,
        target=tvm.target.Target(target, host=target),
    )
    source = compiled_test_mods[0].executor_factory.lib.imported_modules[0].get_source()
    assert f'__attribute__((section(".rodata.tvm"), aligned({constants_byte_alignment})))' in source


def test_output_tensor_names():
    """Test that the output names generated match those in the model"""
    pytest.importorskip("tflite")

    import os
    import tensorflow as tf
    import tflite.Model

    ifm_shape = (1, 299, 299, 3)
    padding = "VALID"
    strides = (1, 1)
    dilation = (1, 1)
    kernel_shape = (3, 2)

    def create_tflite_graph_two_outs():
        """Create a model with 2 output tensors"""

        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                # Use tf.nn API to create the model
                tf_strides = [1, strides[0], strides[1], 1]
                op = tf.nn.conv2d(
                    x,
                    filters=tf.constant(
                        np.random.uniform(size=[kernel_shape[0], kernel_shape[1], 3, 3]),
                        dtype=tf.float32,
                    ),
                    strides=tf_strides,
                    padding=padding,
                    dilations=dilation,
                )
                op = tf.nn.relu(op)
                # Second convolution
                op2 = tf.nn.conv2d(
                    x,
                    filters=tf.constant(
                        np.random.uniform(size=(kernel_shape[0], kernel_shape[1], 3, 3)),
                        dtype=tf.float32,
                    ),
                    strides=strides,
                    padding=padding,
                    data_format="NHWC",
                    dilations=dilation,
                )
                op2 = tf.nn.relu(op2)
                return op, op2

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph_two_outs()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": "int8"},
    )

    use_unpacked_api = True
    interface_api = "c"
    test_runner = AOT_DEFAULT_RUNNER

    in_min, in_max = (-128, 127)
    data = np.random.randint(in_min, high=in_max, size=ifm_shape, dtype="int8")
    inputs = {"x_int8": data}
    output_list = generate_ref_data(mod, inputs, params)
    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        test_runner,
        interface_api,
        use_unpacked_api,
    )

    compiled_test_mods = compile_models(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list, params=params),
        interface_api,
        use_unpacked_api,
    )

    # Check that the names of the output tensors occur in the source code
    source = compiled_test_mods[0].executor_factory.lib.get_source()
    for output_name in output_list.keys():
        assert output_name in source


@pytest.mark.parametrize(
    "workspace_byte_alignment,main_workspace_size",
    [
        (8, 14880),
        (16, 14880),
        (256, 15616),
    ],
)
def test_workspace_calculation(workspace_byte_alignment, main_workspace_size):
    mod, params = tvm.relay.testing.synthetic.get_workload()
    target = "c"
    runtime = Runtime("crt")
    executor = Executor(
        "aot",
        {
            "workspace-byte-alignment": workspace_byte_alignment,
        },
    )
    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.disable_vectorize": True,
        },
    ):
        lib = tvm.relay.build(mod, target, executor=executor, runtime=runtime, params=params)

    mlf_memory_map = mlf._build_function_memory_map(lib.function_metadata)
    assert mlf_memory_map["main"][0]["workspace_size_bytes"] == main_workspace_size


@tvm.testing.requires_package("tflite")
@tvm.testing.requires_cmsisnn
def test_workspace_calculation_cmsis_nn():
    """This tests cmsis_nn codegen for workspace calculation.
    This is tested specially because cmsis-nn codegen creates
    multiple PrimFuncs per offloaded relay function in a non
    -hierarchical manner."""
    pytest.importorskip("tflite")

    from tvm.relay.op.contrib import cmsisnn
    from tvm.contrib.download import download_testdata

    target = "c"
    runtime = Runtime("crt")
    executor = Executor(
        "aot",
        {
            "workspace-byte-alignment": 16,
            "interface-api": "c",
            "unpacked-api": True,
        },
    )

    base_url = "https://github.com/ARM-software/ML-zoo/raw/48a22ee22325d15d2371a6df24eb7d67e21dcc97/models/keyword_spotting/cnn_small/tflite_int8"
    file_to_download = "cnn_s_quantized.tflite"
    file_saved = "cnn_s_quantized_15Dec2021.tflite"
    model_file = download_testdata("{}/{}".format(base_url, file_to_download), file_saved)
    mod, _, params = create_relay_module_and_inputs_from_tflite_file(model_file)
    mod = cmsisnn.partition_for_cmsisnn(mod, params)
    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.disable_vectorize": True,
        },
    ):
        lib = tvm.relay.build(mod, target, executor=executor, runtime=runtime, params=params)
    mlf_memory_map = mlf._build_function_memory_map(lib.function_metadata)
    assert mlf_memory_map["main"][0]["workspace_size_bytes"] == 9904


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
