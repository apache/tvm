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
"""Add an ANEURALNETWORKS_GROUPED_CONV_2D operation with checking
"""
from .error import *


def add_operation(converter, inputs, outputs):
    """Add an ANEURALNETWORKS_GROUPED_CONV_2D operation with checking

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
    assert_nnapi_op_check(len(inputs) == 12)
    ins = [{} for i in range(len(inputs))]

    # check inputs[0]
    ins[0] = {}
    ins[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[0])
    assert_nnapi_op_check(
        ins[0]["dtype"] == "TENSOR_FLOAT32" or ins[0]["dtype"] == "TENSOR_FLOAT16"
    )
    ins[0]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[0])
    assert_nnapi_op_check(ins[0]["rank"] == 4)
    ins[0]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[0])

    # check inputs[1]
    ins[1] = {}
    ins[1]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[1])
    assert_nnapi_op_check(
        ins[1]["dtype"] == "TENSOR_FLOAT32" or ins[1]["dtype"] == "TENSOR_FLOAT16"
    )
    assert_nnapi_op_check(ins[1]["dtype"] == ins[0]["dtype"])
    ins[1]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[1])
    assert_nnapi_op_check(ins[1]["rank"] == 4)
    ins[1]["shape"] = converter.export_obj.helper.operand.get_shape(inputs[1])
    felter = dict(zip(["do", "fh", "fw", "dg"], ins[1]["shape"]))

    # check inputs[2]
    ins[2] = {}
    ins[2]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[2])
    assert_nnapi_op_check(ins[2]["dtype"] == ins[1]["dtype"] and ins[2]["dtype"] == ins[0]["dtype"])
    ins[2]["rank"] = converter.export_obj.helper.operand.get_rank(inputs[2])
    assert_nnapi_op_check(ins[2]["rank"] == 1)
    ins[2]["constant"] = converter.export_obj.helper.operand.get_constant(inputs[2])
    assert_nnapi_op_check(
        ins[2]["constant"]["type"] == "array" and len(ins[2]["constant"]["value"]) == felter["do"]
    )

    # check inputs[3]
    ins[3] = {}
    ins[3]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[3])
    assert_nnapi_op_check(ins[3]["dtype"] == "INT32")
    ins[3]["value"] = converter.export_obj.helper.operand.get_value(inputs[3])
    assert_nnapi_op_check(ins[3]["value"] >= 0)
    padding = {}
    padding["l"] = ins[3]["value"]

    # check inputs[4]
    ins[4] = {}
    ins[4]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[4])
    assert_nnapi_op_check(ins[4]["dtype"] == "INT32")
    ins[4]["value"] = converter.export_obj.helper.operand.get_value(inputs[4])
    assert_nnapi_op_check(ins[4]["value"] >= 0)
    padding["r"] = ins[4]["value"]

    # check inputs[5]
    ins[5] = {}
    ins[5]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[5])
    assert_nnapi_op_check(ins[5]["dtype"] == "INT32")
    ins[5]["value"] = converter.export_obj.helper.operand.get_value(inputs[5])
    assert_nnapi_op_check(ins[5]["value"] >= 0)
    padding["t"] = ins[5]["value"]

    # check inputs[6]
    ins[6] = {}
    ins[6]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[6])
    assert_nnapi_op_check(ins[6]["dtype"] == "INT32")
    ins[6]["value"] = converter.export_obj.helper.operand.get_value(inputs[6])
    assert_nnapi_op_check(ins[6]["value"] >= 0)
    padding["b"] = ins[6]["value"]

    # check inputs[7]
    ins[7] = {}
    ins[7]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[7])
    assert_nnapi_op_check(ins[7]["dtype"] == "INT32")
    ins[7]["value"] = converter.export_obj.helper.operand.get_value(inputs[7])
    assert_nnapi_op_check(ins[7]["value"] >= 0)
    stride = {}
    stride["w"] = ins[7]["value"]

    # check inputs[8]
    ins[8] = {}
    ins[8]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[8])
    assert_nnapi_op_check(ins[8]["dtype"] == "INT32")
    ins[8]["value"] = converter.export_obj.helper.operand.get_value(inputs[8])
    assert_nnapi_op_check(ins[8]["value"] >= 0)
    stride["h"] = ins[8]["value"]

    # check inputs[9]
    ins[9] = {}
    ins[9]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[9])
    assert_nnapi_op_check(ins[9]["dtype"] == "INT32")
    ins[9]["value"] = converter.export_obj.helper.operand.get_value(inputs[9])
    num_groups = ins[9]["value"]
    assert_nnapi_op_check(num_groups >= 0)
    assert_nnapi_op_check(felter["do"] % num_groups == 0)

    # check inputs[10]
    assert_nnapi_op_check(converter.export_obj.helper.operand.is_FuseCode(inputs[10]))

    # check inputs[11]
    ins[11] = {}
    ins[11]["dtype"] = converter.export_obj.helper.operand.get_dtype(inputs[11])
    assert_nnapi_op_check(ins[11]["dtype"] == "BOOL")
    ins[11]["value"] = converter.export_obj.helper.operand.get_value(inputs[11])
    assert_nnapi_op_check(ins[11]["value"] == "false" or ins[11]["value"] == "true")

    # check shapes
    if api_level >= 29 and ins[11]["value"] == "true":
        data_shape = {
            "n": ins[0]["shape"][0],
            "c": ins[0]["shape"][1],
            "h": ins[0]["shape"][2],
            "w": ins[0]["shape"][3],
        }
    else:
        data_shape = {
            "n": ins[0]["shape"][0],
            "h": ins[0]["shape"][1],
            "w": ins[0]["shape"][2],
            "c": ins[0]["shape"][3],
        }

    assert_nnapi_op_check(data_shape["c"] == num_groups * felter["dg"])

    # check outputs
    assert_nnapi_op_check(len(outputs) == 1)
    outs = [{}]

    # check outputs[0]
    outs[0] = {}
    outs[0]["dtype"] = converter.export_obj.helper.operand.get_dtype(outputs[0])
    assert_nnapi_op_check(
        outs[0]["dtype"] == ins[0]["dtype"] and outs[0]["dtype"] == ins[1]["dtype"]
    )
    outs[0]["shape"] = converter.export_obj.helper.operand.get_shape(outputs[0])

    if api_level >= 29 and ins[11]["value"] == "true":
        out_data_shape = {
            "n": outs[0]["shape"][0],
            "c": outs[0]["shape"][1],
            "h": outs[0]["shape"][2],
            "w": outs[0]["shape"][3],
        }
    else:
        out_data_shape = {
            "n": outs[0]["shape"][0],
            "h": outs[0]["shape"][1],
            "w": outs[0]["shape"][2],
            "c": outs[0]["shape"][3],
        }
    total_h = data_shape["h"] + padding["t"] + padding["b"]
    total_w = data_shape["w"] + padding["l"] + padding["r"]
    assert_nnapi_op_check(out_data_shape["n"] == data_shape["n"])
    assert_nnapi_op_check(out_data_shape["h"] == ((total_h - felter["fh"]) // stride["h"] + 1))
    assert_nnapi_op_check(out_data_shape["w"] == ((total_w - felter["fw"]) // stride["w"] + 1))
    assert_nnapi_op_check(out_data_shape["c"] == felter["do"])

    converter.export_obj.add_operation("GROUPED_CONV_2D", inputs, outputs)
