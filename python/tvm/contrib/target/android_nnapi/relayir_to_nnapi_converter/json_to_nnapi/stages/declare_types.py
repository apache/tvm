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
"""Declare and define Android NNAPI ANeuralNetworksOperandType
"""
from .. import templates


def declare_types(lines, export_obj, options):  # pylint: disable=unused-argument
    """Declare and define Android NNAPI ANeuralNetworksOperandType"""
    for t in export_obj["types"]:
        tipe = {
            "name": t["name"],
            "type": templates.ANN_PREFIX + t["type"],
        }
        if "shape" in t:
            tipe["dim_name"] = tipe["name"] + "_dims"
            tipe["shape"] = {
                "rank": len(t["shape"]),
                "str": "{" + ", ".join([str(i) for i in t["shape"]]) + "}",
            }
        lines["tmp"]["model_creation"].append(templates.declare_type.substitute(tipe=tipe))
    return lines, export_obj
