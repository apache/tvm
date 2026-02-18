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
"""Apple Metal GPU target tags."""
from .registry import register_tag


def _register_metal_tag(name, max_threads, shared_mem, warp_size):
    register_tag(
        name,
        {
            "kind": "metal",
            "max_threads_per_block": max_threads,
            "max_shared_memory_per_block": shared_mem,
            "thread_warp_size": warp_size,
            "host": {
                "kind": "llvm",
                "mtriple": "arm64-apple-macos",
                "mcpu": "apple-m4",
            },
        },
    )


_register_metal_tag("apple/m1-gpu", 1024, 32768, 32)
_register_metal_tag("apple/m1-gpu-restricted", 256, 32768, 32)
_register_metal_tag("apple/m2-gpu", 1024, 32768, 32)
