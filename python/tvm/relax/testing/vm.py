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
# pylint: disable=invalid-name
"""Testing utilities for relax VM"""
from typing import Any, List
import numpy as np  # type: ignore

import tvm
from tvm import relax
from tvm.runtime.object import Object


@tvm.register_func("test.vm.move")
def move(src):
    return src


@tvm.register_func("test.vm.add")
def add(a, b):
    ret = a.numpy() + b.numpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.mul")
def mul(a, b):
    ret = a.numpy() * b.numpy()
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.equal_zero")
def equal_zero(a):
    ret = np.all((a.numpy() == 0))
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.subtract_one")
def subtract_one(a):
    ret = np.subtract(a.numpy(), 1)
    return tvm.nd.array(ret)


@tvm.register_func("test.vm.identity")
def identity_packed(a, b):
    b[:] = tvm.nd.array(a.numpy())


@tvm.register_func("test.vm.tile")
def tile_packed(a, b):
    b[:] = tvm.nd.array(np.tile(a.numpy(), (1, 2)))


@tvm.register_func("test.vm.add_scalar")
def add_scalar(a, b):
    return a + b


@tvm.register_func("test.vm.get_device_id")
def get_device_id(device):
    return device.device_id


def check_saved_func(vm: relax.VirtualMachine, func_name: str, *inputs: List[Any]) -> Object:
    # uses save_function to create a closure with the given inputs
    # and ensure the result is the same
    # (assumes the functions return tensors and that they're idempotent)
    saved_name = f"{func_name}_saved"
    vm.save_function(func_name, saved_name, *inputs)
    res1 = vm[func_name](*inputs)
    res2 = vm[saved_name]()
    tvm.testing.assert_allclose(res1.numpy(), res2.numpy(), rtol=1e-7, atol=1e-7)
    return res1


@tvm.register_func("test.vm.check_if_defined")
def check_if_defined(obj: tvm.Object) -> tvm.tir.IntImm:
    return tvm.runtime.convert(obj is not None)
