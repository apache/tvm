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
"""Declare and define C constants used to set operand values."""
from .. import templates

C_TYPES_MAP = {
    "int32": "int32_t",
    "uint32": "uint32_t",
    "float16": "uint16_t",
    "float32": "float",
    "bool": "bool",
}


def declare_constants(lines, export_obj, options):  # pylint: disable=unused-argument
    """Declare and define C constants used to set operand values."""
    for c in export_obj["constants"]:
        tipe = c["type"]
        c_dtype = C_TYPES_MAP[c["dtype"]]
        if tipe == "scalar":
            data = {
                "dtype": c_dtype,
                "name": c["name"],
                "value": c["value"],
            }
        elif tipe == "array":
            data = {
                "dtype": c_dtype,
                "name": c["name"],
                "length": len(c["value"]),
                "value": "{" + ", ".join([str(v) for v in c["value"]]) + "}",
            }
        else:
            raise RuntimeError("Unknown constant type {}".format(tipe))
        lines["tmp"]["model_creation"].append(templates.declare_constant[tipe].substitute(**data))
    return lines, export_obj
