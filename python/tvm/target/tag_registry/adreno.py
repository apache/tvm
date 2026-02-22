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
"""Qualcomm Adreno GPU target tags."""

from .registry import register_tag

register_tag(
    "qcom/adreno-opencl",
    {
        "kind": "opencl",
        "device": "adreno",
        "keys": ["adreno", "opencl", "gpu"],
    },
)

register_tag(
    "qcom/adreno-opencl-clml",
    {
        "kind": "opencl",
        "device": "adreno",
        "keys": ["adreno", "opencl", "gpu", "clml"],
    },
)

register_tag(
    "qcom/adreno-opencl-texture",
    {
        "kind": "opencl",
        "device": "adreno",
        "keys": ["adreno", "opencl", "gpu", "texture"],
    },
)

register_tag(
    "qcom/adreno-vulkan",
    {
        "kind": "vulkan",
        "device": "adreno",
        "keys": ["adreno", "vulkan", "gpu"],
    },
)

register_tag(
    "qcom/adreno-vulkan-texture",
    {
        "kind": "vulkan",
        "device": "adreno",
        "keys": ["adreno", "vulkan", "gpu", "texture"],
    },
)
