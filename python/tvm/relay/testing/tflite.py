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
TensorFlow Lite model generation infrastructure that uses flatbuffers
============================================================
"""
import json
import subprocess
import tempfile
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union
import numpy as np
from tvm.contrib.download import download

# We are currently using TensorFlow Lite 2.4.2 schema to write the model buffers
SCHEMA_URL = (
    "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.4.2/"
    "tensorflow/lite/schema/schema.fbs"
)


class ActivationFunction(Enum):
    NONE = "NONE"
    RELU = "RELU"
    RELU_N1_TO_1 = "RELU_N1_TO_1"
    RELU6 = "RELU6"
    TANH = "TANH"
    SIGN_BIT = "SIGN_BIT"


class Quantization:
    "A class representing quantization of a tensor"

    def __init__(
        self,
        scale: List[float],
        zero_point: List[int],
        quantized_dimension: int = 0,
    ):
        """
        Parameters
        ----------
        scale: List[float]
            The scale(s)
        zero_point: List[int]
            The zero point(s)
        quantized_dimension: int
            The dimension across which quantization is applied
        """
        self.scale = scale
        self.zero_point = zero_point
        self.quantized_dimension = quantized_dimension

    def to_json(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "zero_point": self.zero_point,
            "quantized_dimension": self.quantized_dimension,
        }


class Tensor:
    """A class representing a tensor"""

    def __init__(
        self,
        data_type: str,
        shape: List[int],
        quantization: Optional[Quantization] = None,
        buffer_data: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        data_type: str
            The data type of data in the tensor
        shape: List[int]
            The shape of the tensor
        quantization: Optional[Quantization]
            The quantization parameters of the tensor
        buffer_data: Optional[List[int]]
            The data in the tensor
        """
        self.data_type = data_type
        self.buffer_idx = None
        self.name = None
        self.shape = shape
        self.quantization = quantization
        self.buffer_data = buffer_data

    def to_json(self) -> Dict[str, Any]:
        tensor_json = {
            "type": self.data_type.upper(),
            "buffer": self.buffer_idx,
            "name": self.name,
            "shape": self.shape,
        }
        if self.quantization is not None:
            tensor_json["quantization"] = self.quantization.to_json()
        return tensor_json


class Operator:
    """A class representing an operator"""

    def __init__(
        self,
        opcode: int,
        options_type: str,
        options: Dict[str, Any],
    ):
        """
        Parameters
        ----------
        opcode: int
            The operator's builtin_code
        options_type: str
            The operator's builtin_options_type
        options: Dict[str, Any]
            The operator's builtin_options
        """
        self.opcode = opcode
        self.options_type = options_type
        self.options = options
        self.op_inputs_idx = []
        self.op_outputs_idx = []


def generate_tflite_model(
    inputs: List[Tensor],
    outputs: List[Tensor],
    operator: Operator,
) -> bytes:
    """Generate a TensorFlow Lite model

    Parameters
    ----------
    inputs: List[Tensor],
        The list of input tensors
    outputs: List[Tensor],
        The list of output tensors
    operator: Operator,
        The operator in the model

    Returns
    ------------
    TensorFlow Lite model as bytes
    """
    tmp_dir = tempfile.gettempdir()

    schema_path = tmp_dir + "/schema.fbs"

    download(SCHEMA_URL, schema_path)

    json_path = tmp_dir + "/tflite_model.json"
    tflite_model_path = tmp_dir + "/tflite_model.tflite"

    # figure out which input tensors are inputs to the model and which are inputs to the op
    model_inputs_idx = []

    for idx, tensor in enumerate(inputs):
        # all input tensors are inputs to the operator
        operator.op_inputs_idx.append(idx)
        if tensor.buffer_data is None:
            model_inputs_idx.append(idx)

    tensors = inputs + outputs
    # model and operator has the same output tensors
    model_outputs_idx = list(range(len(inputs), len(tensors)))
    operator.op_outputs_idx = model_outputs_idx

    model_json = _make_json(tensors, operator, model_inputs_idx, model_outputs_idx)
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    subprocess.run(
        ["flatc", "-b", schema_path, json_path],
        cwd=tmp_dir,
        check=True,
    )

    with open(tflite_model_path, "rb") as file:
        model = file.read()
    return model


def _make_json(
    tensors: List[int],
    operator: Operator,
    model_inputs_idx: List[int],
    model_outputs_idx: List[int],
) -> str:

    # first element in list of buffers is always an empty list
    buffers = [{"data": []}]

    # turn the Tensor objects into JSONable dicts
    tensors_as_json = []
    for idx, tensor in enumerate(tensors, start=1):
        tensor.buffer_idx = idx
        tensor.name = "x-" + str(idx)
        tensors_as_json.append(tensor.to_json())

        buffers.append({"data": tensor.buffer_data if tensor.buffer_data else []})

    op = {
        "opcode_index": 0,
        "inputs": operator.op_inputs_idx,
        "outputs": operator.op_outputs_idx,
        "mutating_variable_inputs": [],
    }
    if operator.options_type != "":
        op["builtin_options_type"] = operator.options_type
        op["builtin_options"] = operator.options

    dictionary = {
        "version": 3,
        "operator_codes": [{"builtin_code": operator.opcode}],
        "subgraphs": [
            {
                "tensors": tensors_as_json,
                "inputs": model_inputs_idx,
                "outputs": model_outputs_idx,
                "operators": [op],
            }
        ],
        "buffers": buffers,
    }

    return json.dumps(dictionary, indent=True)


def make_buffer_data(data_type: str, data_low: int, data_high: int, shape: List[int]) -> List[int]:
    """
    Create random data for constant tensors.

    Parameters
    ----------
    data_type : str
        a type string (e.g., int8)
    data_low : int
        smallest value in the tensor
    data_high : int
        highest value in the tensor
    shape : List[int]
        Shape of the tensor to be filled

    Returns
    -------
    data_uint8.tolist() : List[int]
        Buffer data in uint8
    """
    shape_multiplier = np.prod(shape)
    data = np.random.randint(data_low, high=data_high, size=[shape_multiplier], dtype=data_type)
    # The buffer entries in JSON need to be in uint8, so temporarily converting the data
    data_bytes = data.tobytes()
    data_uint8 = np.frombuffer(data_bytes, dtype="uint8")
    return data_uint8.tolist()


def get_range_for_dtype_str(dtype: str) -> Tuple[int, int]:
    """
    Produce the min and max for a give data type.

    Parameters
    ----------
    dtype : str
        a type string (e.g., int8)

    Returns
    -------
    type_info.min : int
        the minimum of the range
    type_info.max : int
        the maximum of the range
    """

    try:
        type_info = np.iinfo(dtype)
    except ValueError:
        type_info = np.finfo(dtype)
    return type_info.min, type_info.max


def get_output_qnn_params(
    weight_shape: List[int],
    input_scale: float,
    input_zp: int,
    weights_scale: Union[float, List[float]],
    weights_zp: int,
    is_depthwise: bool = False,
    input_dtype: str = "int8",
    weights_dtype: str = "int8",
    output_dtype: str = "int8",
) -> Tuple[float, int]:
    """
    Calculate the output quantization parameters for convolution based on the input and
    weights quantization paramters and the data types.

    Parameters
    ----------
    weight_shape : List[int]
        shape of the weights
    input_scale : float
        scale of the input tensor
    input_zp : int
        zero point of the input tensor
    weights_scale : Union[float, List[float]]
        scale(s) of the weights tensor
    weights_zp : int
        zero point of the weights tensor
    is_depthwise : bool
        whether it is a depthwise convolution
    input_dtype : str
        data type of the input tensor
    weights_dtype : str
        data type of the weights tensor
    output_dtype : str
        data type of the output tensor

    Returns
    -------
    output_scale : float
        scale of the output tensor
    output_zp : int
        zero point of the output tensor
    """
    input_dtype_min, input_dtype_max = get_range_for_dtype_str(input_dtype)
    input_max = input_scale * (input_dtype_max - input_zp)
    input_min = input_scale * (input_dtype_min - input_zp)

    weights_dtype_min, weights_dtype_max = get_range_for_dtype_str(weights_dtype)
    weights_sc_max = np.max(weights_scale)
    weights_max = weights_sc_max * (weights_dtype_max - weights_zp)

    weights_sc_min = np.min(weights_scale)
    weights_min = weights_sc_min * (weights_dtype_min - weights_zp)

    weights_h = weight_shape[1]
    weights_w = weight_shape[2]
    channels = weight_shape[3]
    num_elements = weights_h * weights_w * channels
    # Adjust the result if it is a depthwise convolution
    if is_depthwise:
        num_elements = num_elements / channels

    # The smallest and largest possible values in the unquantized output tensor
    output_limits = [
        weights_max * input_max * num_elements,
        weights_min * input_max * num_elements,
        weights_min * input_min * num_elements,
        weights_max * input_min * num_elements,
    ]

    output_max = max(output_limits)
    output_min = min(output_limits)
    output_dtype_min, output_dtype_max = get_range_for_dtype_str(output_dtype)

    output_scale = (output_max - output_min) / (output_dtype_max - output_dtype_min)
    output_zp = int(output_dtype_min - (output_min / output_scale))

    return output_scale, output_zp


class Conv2DOperator(Operator):
    """A class representing a 2D convolution operator"""

    def __init__(
        self,
        padding: str,
        stride_w: int,
        stride_h: int,
        fused_activation_function: ActivationFunction,
        dilation_w: int,
        dilation_h: int,
    ):
        """
        Parameters
        ----------
        padding: str
            The padding, either "SAME" or "VALID"
        stride_w: int
            The stride value on width axis
        stride_h: int
            The stride value on height axis
        fused_activation_function: ActivationFunction
            The fused activation function of the operator
        dilation_w: int
            The dilation value on width axis
        dilation_h: int
            The dilation value on height axis
        """
        options = {
            "padding": padding,
            "stride_w": stride_w,
            "stride_h": stride_h,
            "fused_activation_function": fused_activation_function.value,
            "dilation_w_factor": dilation_w,
            "dilation_h_factor": dilation_h,
        }
        Operator.__init__(
            self,
            opcode=3,
            options_type="Conv2DOptions",
            options=options,
        )


class DepthwiseConv2DOperator(Operator):
    """A class representing a 2D depthwise convolution operator"""

    def __init__(
        self,
        padding: str,
        stride_w: int,
        stride_h: int,
        fused_activation_function: ActivationFunction,
        dilation_w: int,
        dilation_h: int,
        depth_multiplier: int = 1,
    ):
        """
        Parameters
        ----------
        padding: str
            The padding, either "SAME" or "VALID"
        stride_w: int
            The stride value on width axis
        stride_h: int
            The stride value on height axis
        fused_activation_function: ActivationFunction
            The fused activation function of the operator
        dilation_w: int
            The dilation value on width axis
        dilation_h: int
            The dilation value on height axis
        depth_multiplier: int
            The depth multiplier
        """
        options = {
            "padding": padding,
            "stride_w": stride_w,
            "stride_h": stride_h,
            "fused_activation_function": fused_activation_function.value,
            "dilation_w_factor": dilation_w,
            "dilation_h_factor": dilation_h,
            "depth_multiplier": depth_multiplier,
        }
        Operator.__init__(
            self,
            opcode=4,
            options_type="DepthwiseConv2DOptions",
            options=options,
        )
