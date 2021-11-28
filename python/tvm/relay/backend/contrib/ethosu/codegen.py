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
"""Codegen for Arm(R) Ethos(TM)-U NPU"""

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.relay.backend.contrib.ethosu.legalize import LegalizeEthosU
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator
from tvm.relay.backend.contrib.ethosu import util


@tvm._ffi.register_func("relay.ext.ethos-u.constant_updater")
def constant_updater(expr, symbol):  # pylint: disable=unused-argument
    """
    The constant updater process happen after lowering in the core compiler.
    For the NPU, we dont want the build process to extract constants to be loaded in
    the runtime as we are embedding them inside the C runtime.Module.
    """
    return dict()


@tvm._ffi.register_func("relay.ext.ethos-u.relay_to_tir_func")
def relay_to_tir_func(ext_func: relay.Function) -> tvm.tir.PrimFunc:
    """
    This is hook for python-based lowering of relay function
    that gets offloaded to the microNPU.

    Parameters
    ----------
    ext_func : relay.Function
        This is the partitioned relay function

    Returns
    -------
    primfunc : tir.PrimFunc
        This returns the scheduled PrimFunc
    """
    assert len(ext_func.params) == 1
    input_size = util.calculate_size_bytes(ext_func.params[0])
    output_size = util.calculate_size_bytes(ext_func.body)
    mod = tvm.IRModule()
    mod["main"] = ext_func
    mod = LegalizeEthosU()(mod)
    mod = relay.transform.InferType()(mod)
    # We are currently using copy_constants scheduler In the long run,
    # this should be a single intelligent and a composite scheduler
    # that can perform scheduling based on user inputs such as
    # scratch memory size.
    tir_mod, params = lower_to_tir(mod["main"], copy_constants())

    for idx in params.keys():
        params[idx] = tvm.nd.array(params[idx])

    primfunc = tir_mod["main"]
    primfunc = primfunc.with_attr("global_symbol", ext_func.attrs["global_symbol"])
    primfunc = primfunc.with_attr("ethos-u.constants", params)
    primfunc = primfunc.with_attr("ethos-u.input_size", input_size)
    primfunc = primfunc.with_attr("ethos-u.output_size", output_size)
    return primfunc


@tvm._ffi.register_func("relay.ext.ethos-u.primfunc_to_artifact")
def primfunc_to_artifact(primfunc: tvm.tir.PrimFunc) -> util.CompilationArtifact:
    """
    This is hook for python-based lowering of TIR PrimFunc
    that has undergone unified optimization to Compilation
    Artifact destined for the microNPU.

    Parameters
    ----------
    primfunc : tir.PrimFunc
        TIR PrimFuncthat has undergone unified optimization

    Returns
    -------
    CompilationArtifact
        This is a structure that holds the binary artifacts
        for the microNPU
    """
    symbol = str(primfunc.attrs["global_symbol"])
    params = primfunc.attrs["ethos-u.constants"]
    input_size = primfunc.attrs["ethos-u.input_size"]
    output_size = primfunc.attrs["ethos-u.output_size"]
    tir_mod = tvm.IRModule()
    tir_mod[symbol] = primfunc

    params_with_int_keys = dict()
    for idx in params.keys():
        params_with_int_keys[int(idx)] = params[idx].numpy()

    cmms, encoded_constants, scratch_size = tir_to_cs_translator.translate(
        tir_mod, params_with_int_keys
    )
    return util.CompilationArtifact(
        cmms, encoded_constants, scratch_size, input_size, output_size, symbol
    )
