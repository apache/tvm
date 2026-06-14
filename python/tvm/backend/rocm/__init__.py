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
"""ROCm-owned TIRx modules."""


def _detect_target_from_device(dev):
    from tvm.target import Target  # pylint: disable=import-outside-toplevel

    return Target(
        {
            "kind": "rocm",
            "mtriple": "amdgcn-amd-amdhsa-hcc",
            "max_shared_memory_per_block": dev.max_shared_memory_per_block,
            "max_threads_per_block": dev.max_threads_per_block,
            "thread_warp_size": dev.warp_size,
        }
    )


def register_backend():
    """Register ROCm-owned Python semantics."""
    from tvm.target.detect_target import register_device_target_detector

    register_device_target_detector("rocm", _detect_target_from_device)
    return None


__all__ = ["register_backend"]
