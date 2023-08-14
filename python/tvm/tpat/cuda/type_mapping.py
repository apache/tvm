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

# type mapping : tvm -> c, used by c++
tvm_to_c_type_mapping = {
    "bool": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "uint8": "uchar",
    "uint32": "int",
    "uint64": "int",
    "float32": "float",
    "float64": "float",
}

# type mapping : python -> trt, used by TensorRT's getOutputDataType
python_to_trt_type_mapping = {
    "bool": "INT32",
    "int32": "INT32",
    "int64": "INT32",
    "uint64": "INT32",
    "uint8": "INT8",
    "float32": "FLOAT",
    "float64": "FLOAT",
}

# type size : trt workspace, sizeof c++ data type
plugin_type_size = {
    "bool": 4,
    "int16": 4,
    "int32": 4,
    "int64": 4,
    "uint8": 1,
    "uint32": 4,
    "uint64": 4,
    "float32": 4,
    "float64": 4,
}

# onnx type, used by CAST operator
# "int32": 6
onnx_type_mapping = {"int64": 7, "bool": 9, "uint32": 12, "uint64": 13}