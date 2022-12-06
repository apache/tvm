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
Helpers and functions related to the build process to generate code for the Gemmini accelerator
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import tvm

from .environment import Environment
from .transform import *
from tvm import relay
from .legalize import LegalizeGemmini


def preprocess_pass(mod):
    """This is the preprocess pass to use the Gemmini accelerator, it groups the

    Args:
        mod (tvm.ir.IRModule): IRModule to preprocess

    Returns:
        tvm.ir.IRModule: preprocessed IRModule
    """

    # First, merge all dw and convs that can be merged!
    pattern = relay.op.contrib.get_pattern_table("gemmini")

    mod = relay.transform.InferType()(mod)
    mod = relay.transform.ConvertLayout({"qnn.conv2d": ["NHWC", "HWIO"]})(mod)
    mod = relay.transform.SimplifyExpr()(mod)
    mod = relay.transform.MergeComposite(pattern)(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.SimplifyExpr()(mod)
    mod = LegalizeGemmini()(mod)
    mod = relay.transform.InferType()(mod)
    return mod


def internal_build_configs(usmp_alg=""):
    """Builds the internal configurations for the build process

    Args:
        usmp_alg (str, optional): Which USMP algorithm to use. Defaults to "".

    Returns:
        dict: configurations
    """
    enable_usmp = False if usmp_alg == "" else True
    pass_list = [
        (0, tvm.tir.transform.StorageFlatten(16)),
        (1, InjectAMVINIntrin()),
        (1, InjectAMVINIntrinTransposed()),
        (1, InjectBMVINIntrin()),
        (1, InjectBMVINIntrinTransposed()),
        (1, InjectCMVOUTIntrin()),
        (1, InjectCMVOUTIntrinTransposed()),
        (1, InjectDMVINIntrin()),
        (1, InjectDMVINIntrinTransposed()),
        (1, InjectCMVINIntrin()),
        (1, InjectCMVINIntrinTransposed()),
        (1, InjectCMVINAccumIntrin()),
        (1, InjectCMVINAccumIntrinTransposed()),
        (1, tvm.tir.transform.CorrectGemminisScratchpadAndAccumulatorPointers()),
        (2, tvm.tir.transform.LowerDeviceStorageAccessInfo()),
        (4, InsertGemminiHeaderOperators()),
        (5, InsertGemminiFenceOperator()),
    ]

    return {
        "tir.add_lower_pass": pass_list,
        "tir.disable_vectorize": True,
        # "tir.CorrectGemminisScratchpadAndAccumulatorPointers": {"dim": env.DIM}
        "tir.usmp.enable": enable_usmp,
        "tir.usmp.algorithm": usmp_alg,
    }


def build_config(usmp_alg="", **kwargs):
    """Creates the PassContext needed by the build process to correctly build the Gemmini operators

    Args:
        usmp_alg (str, optional): Which USMP algorithm to use. Defaults to "".

    Returns:
        tvm.transform.PassContext: PassContext with specific configurations
    """

    config = internal_build_configs(usmp_alg)
    if kwargs.get("config"):
        config.update(kwargs[config])
        del kwargs["config"]

    return tvm.transform.PassContext(config=config, **kwargs)


def lower(*args, **kwargs):
    """Thin wrapper of tvm.lower

    This wrapper automatically applies Gemmini's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.lower : The original TVM's lower function
    """
    pass_ctx = tvm.transform.PassContext.current()
    if not pass_ctx.config.get("add_lower_pass"):
        with build_config():
            return tvm.lower(*args, **kwargs)
    return tvm.lower(*args, **kwargs)


def build(*args, **kwargs):
    """Thin wrapper of tvm.build

    This wrapper automatically applies Gemmini's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.build : The original TVM's build function
    """
    pass_ctx = tvm.transform.PassContext.current()
    if not pass_ctx.config.get("tir.add_lower_pass"):
        with build_config():
            return tvm.build(*args, **kwargs)
    return tvm.build(*args, **kwargs)


# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.instance().scr_scope)
def mem_info_inp_buffer():
    """Creates the information about the local.scratchpad memory node

    Returns:
        node: The corresponding MemoryInfo node
    """
    spec = Environment.instance()
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=spec.inp_bits,
        max_simd_bits=spec.DIM,
        max_num_bits=int(spec.INP_SCR_ROWS * spec.DIM * spec.inp_bits),
        # head_address=tvm.runtime.const(spec.INP_SCR_BASE_ADDRESS, "uint32"),
        head_address=None,
    )


# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.instance().scr_wgt_scope)
def mem_info_wgt_buffer():
    """Creates the information about the local.scratchpad_weight memory node

    Returns:
        node: The corresponding MemoryInfo node
    """
    spec = Environment.instance()
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=spec.wgt_bits,
        max_simd_bits=spec.DIM,
        max_num_bits=int(spec.WGT_SCR_ROWS * spec.DIM * spec.wgt_bits),
        # head_address=tvm.runtime.const(spec.WGT_SCR_BASE_ADDRESS, "uint32"),
        head_address=None,
    )


# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.instance().acc_scope)
def mem_info_acc_buffer():
    """Creates the information about the local.accumulator memory node

    Returns:
        node: The corresponding MemoryInfo node
    """
    spec = Environment.instance()
    return tvm.ir.make_node(
        "MemoryInfo",
        unit_bits=env.inp_bits,
        max_simd_bits=env.DIM,
        max_num_bits=int(env.ACC_ROWS * env.DIM * env.inp_bits),
        # head_address=tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32"),
        head_address=None,
    )
