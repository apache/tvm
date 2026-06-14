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
"""Vulkan-owned backend hooks."""

RUNTIME_LIBS = ("vulkan",)


def _detect_target_from_device(dev):
    from tvm import get_global_func  # pylint: disable=import-outside-toplevel
    from tvm.target import Target  # pylint: disable=import-outside-toplevel

    f_get_target_property = get_global_func("device_api.vulkan.get_target_property")
    return Target(
        {
            "kind": "vulkan",
            "max_threads_per_block": dev.max_threads_per_block,
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "thread_warp_size": dev.warp_size,
            "supports_float16": f_get_target_property(dev, "supports_float16"),
            "supports_int8": f_get_target_property(dev, "supports_int8"),
            "supports_int16": f_get_target_property(dev, "supports_int16"),
            "supports_int64": f_get_target_property(dev, "supports_int64"),
            "supports_8bit_buffer": f_get_target_property(dev, "supports_8bit_buffer"),
            "supports_16bit_buffer": f_get_target_property(dev, "supports_16bit_buffer"),
            "supports_storage_buffer_storage_class": f_get_target_property(
                dev, "supports_storage_buffer_storage_class"
            ),
        }
    )


def register_backend():
    """Register Vulkan-owned Python semantics."""
    from tvm.target.detect_target import register_device_target_detector

    register_device_target_detector("vulkan", _detect_target_from_device)
    return None


__all__ = ["register_backend", "RUNTIME_LIBS"]
