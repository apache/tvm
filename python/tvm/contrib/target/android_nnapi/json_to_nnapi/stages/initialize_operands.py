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
"""Set initialized value to Android NNAPI operands."""
from .. import templates


def initialize_operands(lines, export_obj, options):
    """Set initialized value to Android NNAPI operands."""
    for i, op in enumerate(export_obj["operands"]):
        value = op.get("value", None)
        if value is None:
            continue

        data = {
            "model": options["model"]["name"],
            "op_idx": i,
        }
        if value["type"] == "constant_idx":
            const = export_obj["constants"][value["value"]]
            data["memory_size"] = "sizeof({})".format(const["name"])
            if const["type"] == "scalar":
                data["memory_ptr"] = "&" + const["name"]
            elif const["type"] == "array":
                data["memory_ptr"] = const["name"]
            else:
                raise RuntimeError(
                    "Unknown const type ({}) for operand {}".format(const["type"], i)
                )
            lines["tmp"]["model_creation"].append(
                templates.initialize_operand["memory_ptr"].substitute(**data)
            )
        elif value["type"] == "memory_ptr":
            pass
        elif value["type"] == "ann_memory":
            memory = export_obj["memories"][value["value"]]
            data["memory_idx"] = value["value"]
            data["length"] = memory["size"]
            lines["tmp"]["model_creation"].append(
                templates.initialize_operand["ann_memory"].substitute(**data)
            )
        else:
            raise RuntimeError("Unknown value type ({}) for operand {}".format(value["type"], i))
    return lines, export_obj
