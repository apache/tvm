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
# pylint: disable=wildcard-import,unused-wildcard-import
"""Add an ANEURALNETWORKS_TRANSPOSE operation with checking
"""
from .error import *


def add_operation(converter, inputs, outputs):
    """Add an ANEURALNETWORKS_TRANSPOSE operation with checking

    Parameters
    ----------
    converter: FunctionToJsonConverter
        the converter object holding export_obj

    inputs: list of int
        inputs to the operation

    outputs: list of int
        outputs of the operation

    """
    api_level = converter.options["target"]["api_level"]
    assert_anc_compatibility(
        api_level >= 28,
        f"Target Android API level { api_level } is too low to support the operation",
    )

    # check inputs
    assert_nnapi_op_check(len(inputs) == 2)
    ins = [{}, {}]

    # check inputs[0]
    ins[0] = {}
    ins[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[0])
    if ins[0]["dtype"] == "TENSOR_FLOAT16":
        assert_nnapi_op_check(api_level >= 29)
    else:
        assert_nnapi_op_check(ins[0]["dtype"] == "TENSOR_FLOAT32")
    ins[0]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[0])
    ins[0]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[0])
    assert_nnapi_op_check(ins[0]["rank"] <= 4)

    # check inputs[1]
    ins[1] = {}
    ins[1]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[1])
    assert_nnapi_op_check(ins[1]["dtype"] == "TENSOR_INT32")
    ins[1]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[1])
    assert_nnapi_op_check(ins[1]["rank"] == 1)
    ins[1]["constant"] = converter.export_obj.helper.operand.get_constant(inputs[1])
    assert_nnapi_op_check(
        ins[1]["constant"]["type"] == "array" and len(ins[1]["constant"]["value"]) == ins[0]["rank"]
    )
    ins[1]["value"] = converter.export_obj.helper.operand.get_value(inputs[1])

    # check outputs
    assert_nnapi_op_check(len(outputs) == 1)
    outs = [{}]

    # check outputs[0]
    outs[0] = {}
    outs[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(outputs[0])
    assert_nnapi_op_check(outs[0]["dtype"] == ins[0]["dtype"])
    outs[0]["shape"] = converter.export_obj.helper.operand.get_shape(outputs[0])
    assert_nnapi_op_check(outs[0]["shape"] == [ins[0]["shape"][i] for i in ins[1]["value"]])

    converter.export_obj.add_operation("TRANSPOSE", inputs, outputs)
