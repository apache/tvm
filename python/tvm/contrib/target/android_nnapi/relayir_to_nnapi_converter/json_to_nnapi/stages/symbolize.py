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
"""Prepare JSON object for Android NNAPI codegen
"""


def symbolize(lines, export_obj, options):  # pylint: disable=unused-argument
    """Assign C symbols to JSON objects"""

    def _symbolize_types(types):
        cnts = {
            "tensor": 0,
            "scalar": 0,
        }
        for t in types:
            if t["type"].startswith("TENSOR_"):
                t["name"] = "tensor" + str(cnts["tensor"])
                cnts["tensor"] += 1
            else:
                t["name"] = "scalar" + str(cnts["scalar"])
                cnts["scalar"] += 1

    _symbolize_types(export_obj["types"])

    def _symbolize_consts(consts):
        cnt = 0
        for c in consts:
            c["name"] = "const_val" + str(cnt)
            cnt += 1

    if "constants" in export_obj:
        _symbolize_consts(export_obj["constants"])

    return lines, export_obj
