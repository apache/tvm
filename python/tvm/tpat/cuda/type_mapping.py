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

# type mapping : tvm -> c
tvm_to_c_type_mapping = {
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "float32": "float",
    "uint64": "int",
    "uint8": "int8",
    "uint1": "int",
    "uint32": "int",
    "float64": "float",
    "bool": "int",
}

# type mapping : python -> trt
python_to_trt_type_mapping = {
    "bool": "INT32",
    "int32": "INT32",
    "int64": "INT32",
    "float32": "FLOAT",
    "uint64": "INT32",
    "uint8": "INT8",
    "uint1": "INT32",
    "float64": "FLOAT",
}

# type size : trt workspace
plugin_type_size = {
    "int16": 4,
    "int32": 4,
    "float32": 4,
    "int64": 4,
    "uint32": 4,
    "uint64": 4,
    "uint8": 1,
    "uint1": 1,
    "float64": 4,
}

# onnx type
onnx_type_mapping = {"int64": 7, "bool": 9, "uint32": 12, "uint64": 13}
# "int32": 6