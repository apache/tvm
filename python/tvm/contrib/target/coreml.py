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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""Utility to compile CoreML models"""

import os
import shutil

import tvm._ffi
from ...relay.expr_functor import ExprVisitor
from .. import xcode, coreml_runtime


def _convert_add(builder, name, inputs, outputs, args, attrs):
    builder.add_elementwise(name=name, input_names=inputs, output_name=outputs[0], mode="ADD")


def _convert_multiply(builder, name, inputs, outputs, args, attrs):
    builder.add_elementwise(name=name, input_names=inputs, output_name=outputs[0], mode="MULTIPLY")


def _convert_clip(builder, name, inputs, outputs, args, attrs):
    builder.add_clip(
        name=name,
        input_name=inputs[0],
        output_name=outputs[0],
        min_value=attrs.a_min,
        max_value=attrs.a_max,
    )


def _convert_batch_flatten(builder, name, inputs, outputs, args, attrs):
    builder.add_flatten_to_2d(name=name, input_name=inputs[0], output_name=outputs[0])


def _convert_expand_dims(builder, name, inputs, outputs, args, attrs):
    if attrs.axis >= 0:
        axes = list(range(attrs.axis, attrs.axis + attrs.num_newaxis))
    else:
        axes = list(range(attrs.axis - attrs.num_newaxis + 1, attrs.axis + 1))

    builder.add_expand_dims(name=name, input_name=inputs[0], output_name=outputs[0], axes=axes)


def _convert_relu(builder, name, inputs, outputs, args, attrs):
    builder.add_activation(
        name=name, non_linearity="RELU", input_name=inputs[0], output_name=outputs[0]
    )


def _convert_softmax(builder, name, inputs, outputs, args, attrs):
    builder.add_softmax_nd(
        name=name, input_name=inputs[0], output_name=outputs[0], axis=int(attrs["axis"])
    )


def _convert_conv2d(builder, name, inputs, outputs, args, attrs):
    weight = args[1].data.numpy()
    if attrs["kernel_layout"] == "OIHW":
        # convert to 'HWIO'
        weight = weight.transpose([2, 3, 1, 0])
    kh, kw, kc, oc = weight.shape

    builder.add_convolution(
        name=name,
        kernel_channels=kc,
        output_channels=oc,
        height=kh,
        width=kw,
        stride_height=int(attrs["strides"][0]),
        stride_width=int(attrs["strides"][0]),
        border_mode="valid",
        groups=int(attrs["groups"]),
        W=weight,
        b=None,
        has_bias=False,
        input_name=inputs[0],
        output_name=outputs[0],
        dilation_factors=[int(v) for v in attrs["dilation"]],
        padding_top=int(attrs["padding"][0]),
        padding_bottom=int(attrs["padding"][2]),
        padding_left=int(attrs["padding"][1]),
        padding_right=int(attrs["padding"][3]),
    )


def _convert_global_avg_pool2d(builder, name, inputs, outputs, args, attrs):
    builder.add_pooling(
        name=name,
        height=1,
        width=1,
        stride_height=1,
        stride_width=1,
        layer_type="AVERAGE",
        padding_type="VALID",
        input_name=inputs[0],
        output_name=outputs[0],
        is_global=True,
    )


_convert_map = {
    "add": _convert_add,
    "multiply": _convert_multiply,
    "clip": _convert_clip,
    "expand_dims": _convert_expand_dims,
    "nn.relu": _convert_relu,
    "nn.batch_flatten": _convert_batch_flatten,
    "nn.softmax": _convert_softmax,
    "nn.conv2d": _convert_conv2d,
    "nn.global_avg_pool2d": _convert_global_avg_pool2d,
}


class CodegenCoreML(ExprVisitor):
    """
    A visitor to traverse subgraphs and build Core ML models.
    """

    def __init__(self, model_name, function):
        import coremltools
        from coremltools.models.neural_network import NeuralNetworkBuilder

        ExprVisitor.__init__(self)
        self.model_name = model_name
        self.function = function
        self.out_map = {}
        self.model_inputs_ = []
        self.buf_idx_ = 0

        # Update inputs and outputs after we visit all the nodes.
        # Set dummy values for now.
        # TODO: support multiple outputs
        inputs = [
            (
                "",
                coremltools.models.datatypes.Array(
                    1,
                ),
            )
            for _ in self.function.params
        ]
        outputs = [
            (
                "",
                coremltools.models.datatypes.Array(
                    1,
                ),
            )
        ]
        self.builder = NeuralNetworkBuilder(inputs, outputs, disable_rank5_shape_mapping=True)

    def visit_constant(self, const):
        output = "buf_" + str(self.buf_idx_)
        self.builder.add_load_constant_nd(
            name=output,
            output_name=output,
            constant_value=const.data.numpy(),
            shape=const.data.shape,
        )
        self.buf_idx_ = self.buf_idx_ + 1
        self.out_map[const] = [output]

    def visit_var(self, var):
        name = var.name_hint
        shape = [int(n) for n in var.type_annotation.shape]
        dtype = var.type_annotation.dtype
        self.model_inputs_.append((name, shape, dtype))
        self.out_map[var] = [name]

    def visit_call(self, call):
        inputs = []
        for arg in call.args:
            super().visit(arg)
            for out in self.out_map[arg]:
                inputs.append(out)
        outputs = ["buf_" + str(self.buf_idx_)]
        op_name = call.op.name
        layer_name = op_name + "_" + str(self.buf_idx_)

        assert op_name in _convert_map, "{} is not supported".format(op_name)
        _convert_map[op_name](self.builder, layer_name, inputs, outputs, call.args, call.attrs)

        self.buf_idx_ = self.buf_idx_ + 1
        self.out_map[call] = outputs

    def compile(self, out_dir):
        """
        Build a Core ML model and compile it with Xcode toolchain.
        """
        import coremltools
        from coremltools.proto.Model_pb2 import ArrayFeatureType

        FEATURE_TYPE_MAP = {
            "float32": ArrayFeatureType.FLOAT32,
            "float64": ArrayFeatureType.DOUBLE,
            "int32": ArrayFeatureType.INT32,
        }

        input_names, input_dims, input_dtypes = zip(*self.model_inputs_)
        self.builder.set_input(input_names, input_dims)
        for i, dtype in enumerate(input_dtypes):
            assert dtype in FEATURE_TYPE_MAP
            input_desc = self.builder.spec.description.input
            input_desc[i].type.multiArrayType.dataType = FEATURE_TYPE_MAP[dtype]

        output_dim = [int(n) for n in self.function.ret_type.shape]
        self.builder.set_output(self.out_map[self.function.body], [output_dim])
        for i, dtype in enumerate([self.function.ret_type.dtype]):
            assert dtype in FEATURE_TYPE_MAP
            output_desc = self.builder.spec.description.output
            output_desc[i].type.multiArrayType.dataType = FEATURE_TYPE_MAP[dtype]

        model = coremltools.models.MLModel(self.builder.spec)
        xcode.compile_coreml(model, self.model_name, out_dir)


@tvm._ffi.register_func("relay.ext.coremlcompiler")
def coreml_compiler(func):
    """
    Create a CoreML runtime from a Relay module.
    """
    assert isinstance(func, tvm.relay.function.Function)
    model_dir = os.getcwd()
    name = str(func.attrs.global_symbol)
    builder = CodegenCoreML(name, func)
    builder.visit(func.body)
    mlmodelc_path = "{}/{}.mlmodelc".format(model_dir, name)
    if os.path.exists(mlmodelc_path):
        shutil.rmtree(mlmodelc_path)
    builder.compile(model_dir)

    dev = tvm.cpu(0)
    return coreml_runtime.create(name, mlmodelc_path, dev).module
