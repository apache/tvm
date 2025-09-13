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
"""
Example NPU Backend for BYOC Integration

This module provides an educational example of how to implement
a custom NPU backend in TVM using the Bring Your Own Codegen (BYOC)
framework. It demonstrates key NPU architectural concepts including
memory hierarchy, tiling, quantization, and operation fusion.

The patterns module registers all supported NPU operations and their
constraints, making them available for graph partitioning.
"""

from . import patterns  # noqa: F401

__all__ = ["patterns"]
