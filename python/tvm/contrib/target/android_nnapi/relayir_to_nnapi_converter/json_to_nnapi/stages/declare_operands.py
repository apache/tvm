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
"""Declare Android NNAPI operands."""
from .. import templates


def declare_operands(lines, export_obj, options):
    """Declare Android NNAPI operands."""
    for i, op in enumerate(export_obj["operands"]):
        op_type = export_obj["types"][op["type"]]
        data = {
            "model": options["model"]["name"],
            "type": op_type["name"],
            "index": i,
        }
        lines["tmp"]["model_creation"].append(templates.declare_operand.substitute(**data))
    return lines, export_obj
