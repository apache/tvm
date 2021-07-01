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
"""Specify Android NNAPI model input and output operands."""
from .. import templates


def declare_inputs_outputs(lines, export_obj, options):
    """Specify Android NNAPI model input and output operands."""
    inputs = export_obj["inputs"]
    outputs = export_obj["outputs"]
    data = {
        "inputs": {
            "length": len(inputs),
            "str": "{" + ", ".join([str(i) for i in inputs]) + "}",
        },
        "outputs": {
            "length": len(outputs),
            "str": "{" + ", ".join([str(i) for i in outputs]) + "}",
        },
        "model": options["model"]["name"],
    }
    lines["tmp"]["model_creation"].append(templates.declare_inputs_outputs.substitute(**data))
    return lines, export_obj
