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
# pylint: disable=invalid-name, dangerous-default-value, arguments-differ
"""Driver for partitioning and building a Relay module for ComposableKernel offload."""
import os


def get_composable_kernel_path():
    invalid_paths = []
    for rel in ["../../../../", "../../../", "../../"]:
        tvm_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel)
        composable_kernel_path = os.path.join(tvm_root, "3rdparty/composable_kernel")
        if os.path.exists(composable_kernel_path):
            return composable_kernel_path
        invalid_paths.append(composable_kernel_path)
    raise AssertionError(f"The ComposableKernel root directory not found in: {invalid_paths}")
