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
"""Codegen for Scale4Edge NPUs"""

import tvm
from tvm import relay, te, tir


@tvm._ffi.register_func("relay.ext.generic.relay_to_tir_func")
def relay_to_tir_func(ext_func: relay.Function) -> tvm.tir.PrimFunc:
    """
    This is the hook for python-based lowering of relay function
    that gets offloaded to the target NPU.

    Parameters
    ----------
    ext_func : relay.Function
        This is the partitioned relay function

    Returns
    -------
    primfunc : tir.PrimFunc
        This returns the scheduled PrimFunc
    """
    f = tvm._ffi.get_global_func("relay.backend.LowerToTE")
    te_func = f(ext_func)
    primfunc = te.create_prim_func_from_outputs(te_func.outputs)
    primfunc = primfunc.with_attr("global_symbol", ext_func.attrs["global_symbol"])

    mod = tvm.IRModule()
    mod["main"] = primfunc
    mod = tir.transform.StorageFlatten(64, False)(mod)
    mod = tir.transform.LowerInitBlock()(mod)
    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tir.transform.CompactBufferAllocation()(mod)
    mod = tir.transform.LowerMatchBuffer()(mod)
    mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.Simplify()(mod)

    return mod["main"]
