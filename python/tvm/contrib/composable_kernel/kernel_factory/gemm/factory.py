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
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import, pointless-exception-statement

"""Database implementation for managing ComposableKernel kernels."""
from dataclasses import dataclass
from typing import List

from ... import library
from . import library as gemm


@dataclass(eq=True, frozen=True)
class DeviceGemmInstanceKey:
    a_dtype: library.DataType
    b_dtype: library.DataType
    c_dtype: library.DataType
    a_layout: library.LayoutType
    b_layout: library.LayoutType
    c_layout: library.LayoutType
    batched: bool


class DeviceGemmInstanceRegistry:
    def __init__(self):
        self.kvstore = {}

    def register(self, key: DeviceGemmInstanceKey, instances: List[gemm.GemmOperation]):
        if key not in self.kvstore:
            self.kvstore[key] = []
        self.kvstore[key].extend(instances)

    def get(self, key: DeviceGemmInstanceKey) -> List[gemm.GemmOperation]:
        return self.kvstore.get(key, [])


_device_gemm_instance_factory = DeviceGemmInstanceRegistry()


def register_instances(key: DeviceGemmInstanceKey, instances: List[gemm.GemmOperation]):
    _device_gemm_instance_factory.register(key, instances)


def get_instances(
    a_dtype: library.DataType,
    b_dtype: library.DataType,
    c_dtype: library.DataType,
    a_layout: library.LayoutType,
    b_layout: library.LayoutType,
    c_layout: library.LayoutType,
    batched: bool,
) -> List[gemm.GemmOperation]:
    key = DeviceGemmInstanceKey(a_dtype, b_dtype, c_dtype, a_layout, b_layout, c_layout, batched)
    return _device_gemm_instance_factory.get(key)
