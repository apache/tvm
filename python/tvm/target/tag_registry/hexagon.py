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
"""Qualcomm Hexagon target tags."""

from .registry import register_tag

_ONE_MB = 2**20

_HEXAGON_VERSIONS = {
    "v65": {"vtcm": _ONE_MB // 4, "mattr": ["+hvxv65", "+hvx-length128b"]},
    "v66": {"vtcm": _ONE_MB // 4, "mattr": ["+hvxv66", "+hvx-length128b"]},
    "v68": {
        "vtcm": 4 * _ONE_MB,
        "mattr": ["+hvxv68", "+hvx-length128b", "+hvx-qfloat", "-hvx-ieee-fp"],
        "llvm-options": ["-force-hvx-float"],
    },
    "v69": {
        "vtcm": 8 * _ONE_MB,
        "mattr": ["+hvxv69", "+hvx-length128b", "+hvx-qfloat", "-hvx-ieee-fp"],
    },
    "v73": {
        "vtcm": 8 * _ONE_MB,
        "mattr": ["+hvxv73", "+hvx-length128b", "+hvx-qfloat", "-hvx-ieee-fp"],
    },
    "v75": {
        "vtcm": 8 * _ONE_MB,
        "mattr": ["+hvxv75", "+hvx-length128b", "+hvx-qfloat", "-hvx-ieee-fp"],
    },
}

for _ver, _info in _HEXAGON_VERSIONS.items():
    _config = {
        "kind": "hexagon",
        "mtriple": "hexagon",
        "mcpu": "hexagon" + _ver,
        "mattr": _info["mattr"],
        "num-cores": 4,
        "vtcm-capacity": _info["vtcm"],
    }
    if "llvm-options" in _info:
        _config["llvm-options"] = _info["llvm-options"]
    register_tag("qcom/hexagon-" + _ver, _config)
