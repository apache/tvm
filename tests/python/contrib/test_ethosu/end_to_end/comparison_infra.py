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
"""
This module provides infrastructure to verify the correctness of
the command stream produced.
Currently it will invoke vela to generate a vela-optimized tflite
in which the command stream is contained as a custom operator.
This class include methods to parse the custom operator to extract
the command stream and perform an equivalency check for single operator
test cases.
"""
# pylint: disable=invalid-name, unused-argument
import pytest

pytest.importorskip("ethosu.vela")

import numpy as np
import tflite.Model

import tvm
import tensorflow as tf
from tvm import relay

from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.backend.contrib.ethosu import preprocess

from tvm.relay.op.contrib.ethosu import partition_for_ethosu

from tests.python.contrib.test_ethosu import infra


def _compare_ethosu_with_reference(
    mod, input_data, output_data, accel_type, output_tolerance=0, print_cmm=False
):
    compiled_models = infra.build_source(
        mod,
        input_data,
        output_data,
        accel_type,
        output_tolerance=output_tolerance,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    if print_cmm:
        get_artifacts = tvm._ffi.get_global_func("runtime.module.ethos-u.get_artifacts")
        compilation_artifacts = get_artifacts(ethosu_module)
        cmms = bytes.fromhex(compilation_artifacts[0].command_stream)
        infra.print_payload(cmms)

    infra.verify_source(compiled_models, accel_type)


def _compare_tvm_with_tflite(
    tf_func, shapes, accel_type, ranges=None, output_tolerance=0, print_cmm=False
):
    mod, tflite_graph = _get_tflite_graph(tf_func, shapes, ranges)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

    _compare_ethosu_with_reference(
        mod,
        input_data,
        output_data,
        accel_type,
        output_tolerance=output_tolerance,
        print_cmm=print_cmm,
    )


def _get_tflite_graph(tf_func, shapes, ranges=None):
    tensor_specs = [tf.TensorSpec(shape, dtype=tf.float32) for shape in shapes]
    if not ranges:
        ranges = [(0, 1) for _ in shapes]
    concrete_func = tf_func.get_concrete_function(*tensor_specs)

    # Convert the model
    def representative_dataset():
        for _ in range(100):
            inputs = []
            for i, shape in enumerate(shapes):
                data = np.random.uniform(
                    low=ranges[i][0], high=ranges[i][1], size=tuple(shape)
                ).astype("float32")
                inputs.append(data)

            yield inputs

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_graph = converter.convert()

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(tflite_model)
    mod = partition_for_ethosu(relay_module, params)
    return mod, tflite_graph


class EthosUAnnotator(ExprMutator):
    """Annotate entire graph for Ethos-U offload"""

    def __init__(self):
        super(EthosUAnnotator, self).__init__()
        self.compiler = "ethos-u"
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Var):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = compiler_end(new_call, self.compiler)
        return new_call

    def visit_constant(self, constant):
        new_constant = compiler_begin(constant, self.compiler)
        return new_constant


def _create_ethosu_partition(mod):
    mod["main"] = EthosUAnnotator().visit(mod["main"])
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    mod = preprocess.preprocess_ext_io()(mod)
    return mod
