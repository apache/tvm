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
"""Wraps the Android NNAPI Model in a class
"""
from .. import templates


# NOTICE: make sure TVM maps type A to type B before modifying this table!!
C_TYPES_MAP = {
    "BOOL": "bool",
    "FLOAT32": "float",
    "INT32": "int",
    "TENSOR_BOOL8": "bool",
    "TENSOR_FLOAT16": "uint16_t",
    "TENSOR_FLOAT32": "float",
    "TENSOR_INT32": "int",
}


def declare_wrapper_class(lines, export_obj, options):
    """Wraps the Android NNAPI Model in a class"""
    data = {
        "class": {
            "self": {
                "name": options["class"]["name"],
            },
            "model": {
                "name": options["model"]["name"],
            },
            "compilation": {
                "name": options["compilation"]["name"],
            },
            "execution": {
                "name": options["execution"]["name"],
                "end_event_name": options["execution"]["end_event_name"],
            },
        },
        "codes": {
            "model_creation": "\n".join(
                ["    " + s for s in "\n".join(lines["tmp"]["model_creation"]).split("\n")]
            ),
            "set_execution_io": "\n".join(
                ["    " + s for s in "\n".join(lines["tmp"]["set_execution_io"]).split("\n")]
            ),
        },
    }

    def _scope():
        var_decls = []
        for inp in export_obj["inputs"]:
            op = export_obj["operands"][inp]
            assert op["value"]["type"] == "memory_ptr"
            tipe = export_obj["types"][op["type"]]
            var_decls.append("{}* {}".format(C_TYPES_MAP[tipe["type"]], op["value"]["value"]))
        for outp in export_obj["outputs"]:
            op = export_obj["operands"][outp]
            assert op["value"]["type"] == "memory_ptr"
            tipe = export_obj["types"][op["type"]]
            var_decls.append("{}* {}".format(C_TYPES_MAP[tipe["type"]], op["value"]["value"]))
        data["class"]["execution"]["func_params_decl_str"] = ", ".join(var_decls)

    _scope()
    lines["tmp"]["wrapper_class"].append(templates.declare_wrapper_class.substitute(**data))
    return lines, export_obj
