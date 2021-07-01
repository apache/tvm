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
"""Sets the inputs and outputs for the generated Android NNAPI
model."""
import re
from functools import reduce
from .. import templates


def set_execution_inputs_outputs(lines, export_obj, options):
    """Sets the inputs and outputs for the generated Android NNAPI
    model."""
    for i, op_i in enumerate(export_obj["inputs"]):
        op = export_obj["operands"][op_i]
        value = op["value"]
        assert value["type"] == "memory_ptr"

        data = {
            "execution": options["execution"]["name"],
            "input_idx": i,
        }
        tipe = export_obj["types"][op["type"]]
        nnapi_dtype = tipe["type"]
        nbits = int((lambda s: s if s != "" else "8")(re.sub(r"^[^0-9]+", "", nnapi_dtype)))
        assert (nbits != 0) and (nbits % 8 == 0)
        data["memory_ptr"] = value["value"]
        if nnapi_dtype.startswith("TENSOR"):
            data["memory_size"] = reduce(lambda a, b: a * b, tipe["shape"], 1) * nbits // 8
        else:
            data["memory_size"] = nbits // 8
        lines["tmp"]["set_execution_io"].append(templates.set_execution_input.substitute(**data))

    def _outputs():
        assert len(export_obj["outputs"]) == 1
        op = export_obj["operands"][export_obj["outputs"][0]]
        value = op["value"]
        assert value["type"] == "memory_ptr"

        data = {
            "execution": options["execution"]["name"],
            "output_idx": 0,
        }
        tipe = export_obj["types"][op["type"]]
        nnapi_dtype = tipe["type"]
        nbits = int((lambda s: s if s != "" else "8")(re.sub(r"^[^0-9]+", "", nnapi_dtype)))
        assert (nbits != 0) and (nbits % 8 == 0)
        data["memory_ptr"] = value["value"]
        if nnapi_dtype.startswith("TENSOR"):
            data["memory_size"] = reduce(lambda a, b: a * b, tipe["shape"], 1) * nbits // 8
        else:
            data["memory_size"] = nbits // 8
        lines["tmp"]["set_execution_io"].append(templates.set_execution_output.substitute(**data))

    _outputs()
    return lines, export_obj
