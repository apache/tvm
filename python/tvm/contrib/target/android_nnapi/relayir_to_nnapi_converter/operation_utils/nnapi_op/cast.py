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
"""Add an ANEURALNETWORKS_CAST operation with checking
"""
from .error import *


def add_operation(converter, inputs, outputs):
    """Add an ANEURALNETWORKS_CAST operation with checking

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
        api_level >= 29,
        f"Target Android API level { api_level } is too low to support the operation",
    )

    # check inputs
    assert_nnapi_op_check(len(inputs) == 1)
    ins = [{}]

    # check inputs[0]
    ins[0] = {}
    ins[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[0])
    assert_nnapi_op_check(
        ins[0]["dtype"] == "TENSOR_FLOAT16"
        or ins[0]["dtype"] == "TENSOR_FLOAT32"
        or ins[0]["dtype"] == "TENSOR_INT32"
    )
    ins[0]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[0])

    # check outputs
    assert_nnapi_op_check(len(outputs) == 1)
    outs = [{}]

    # check outputs[0]
    outs[0] = {}
    outs[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(outputs[0])
    assert_nnapi_op_check(
        outs[0]["dtype"] == "TENSOR_FLOAT16"
        or outs[0]["dtype"] == "TENSOR_FLOAT32"
        or outs[0]["dtype"] == "TENSOR_INT32"
    )
    outs[0]["shape"] = converter.export_obj.helper.operand.get_shape(outputs[0])
    assert_nnapi_op_check(outs[0]["shape"] == ins[0]["shape"])

    converter.export_obj.add_operation("CAST", inputs, outputs)
