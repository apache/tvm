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
"""Codegen for Arm(R) Ethos(TM)-U"""
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.relay.backend.contrib.ethosu.legalize import LegalizeEthosU
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator
from tvm.relay.backend.contrib.ethosu import util


@tvm._ffi.register_func("relay.ext.ethosu")
def ethosu_compiler(external_function):
    """The entry-point to a compile a external relay function of
    NPU compatible operators to generated command stream.
    Such generated command stream would be used to create c-source r
    runtime module that interfaces with NPU driver.
    """
    assert isinstance(external_function, tvm.ir.function.BaseFunc)
    func_name = external_function.attrs["global_symbol"]
    # There should only be a single input
    assert len(external_function.params) == 1
    input_size = util.calculate_size_bytes(external_function.params[0])
    output_size = util.calculate_size_bytes(external_function.body)
    cmms, encoded_constants, scratch_size = _compile(external_function)
    ethosu_runtime = tvm._ffi.get_global_func("runtime.module.ethosu.create")
    return ethosu_runtime(func_name, cmms, encoded_constants, scratch_size, input_size, output_size)


@tvm._ffi.register_func("relay.ext.ethosu.constant_updater")
def constant_updater(expr, symbol):  # pylint: disable=unused-argument
    """
    The constant updater process happen after lowering in the core compiler.
    For the NPU, we dont want the build process to extract constants to be loaded in
    the runtime as we are embedding them inside the C runtime.Module.
    """
    return dict()


def _compile(ext_func):
    """
    This is the main wrapper that accepts an external
    relay function and runs all the passes to lower it down
    to command stream
    Parameters
    ----------
    ext_func : tvm.relay.function.Function
        The partitioned relay function
    Returns
    -------
    cs : str
        An hex string of the bytes of command stream
    encoded_constants : str
        An hex string of the bytes that includes concat'd
        encoded weights, encoded biases and scales.
    scratch_size : int
        The size of the scratch buffer needed.
    """
    mod = tvm.IRModule()
    mod["main"] = ext_func
    mod = LegalizeEthosU()(mod)
    mod = relay.transform.InferType()(mod)
    # We are currently using copy_constants scheduler In the long run,
    # this should be a single intelligent and a composite scheduler
    # that can perform scheduling based on user inputs such as
    # scratch memory size.
    tir_mod, params = lower_to_tir(mod["main"], copy_constants())
    cmms, encoded_constants, scratch_size = tir_to_cs_translator.translate(tir_mod, params)
    return cmms, encoded_constants, scratch_size
