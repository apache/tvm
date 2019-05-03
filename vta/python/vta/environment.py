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
"""Configurable VTA Hareware Environment scope."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import os
import json
import copy
import tvm
from . import intrin
from .pkg_config import PkgConfig


class DevContext(object):
    """Internal development context

    This contains all the non-user facing compiler
    internal context that is hold by the Environment.

    Parameters
    ----------
    env : Environment
        The environment hosting the DevContext

    Note
    ----
    This class is introduced so we have a clear separation
    of developer related, and user facing attributes.
    """
    # Memory id for DMA
    MEM_ID_UOP = 0
    MEM_ID_WGT = 1
    MEM_ID_INP = 2
    MEM_ID_ACC = 3
    MEM_ID_OUT = 4
    # VTA ALU Opcodes
    ALU_OPCODE_MIN = 0
    ALU_OPCODE_MAX = 1
    ALU_OPCODE_ADD = 2
    ALU_OPCODE_SHR = 3
    # Task queue id (pipeline stage)
    QID_LOAD_INP = 1
    QID_LOAD_WGT = 1
    QID_LOAD_OUT = 2
    QID_STORE_OUT = 3
    QID_COMPUTE = 2

    def __init__(self, env):
        self.vta_axis = tvm.thread_axis("vta")
        self.vta_push_uop = tvm.make.StringImm("VTAPushGEMMOp")
        ctx = tvm.call_extern("handle", "VTATLSCommandHandle")
        self.command_handle = tvm.make.Call(
            "handle", "tvm_thread_context", [ctx],
            tvm.expr.Call.Intrinsic, None, 0)
        self.DEBUG_NO_SYNC = False
        env._dev_ctx = self
        self.gemm = intrin.gemm(env, env.mock_mode)

    def get_task_qid(self, qid):
        """Get transformed queue index."""
        return 1 if self.DEBUG_NO_SYNC else qid


class Environment(object):
    """Hardware configuration object.

    This object contains all the information
    needed for compiling to a specific VTA backend.

    Parameters
    ----------
    cfg : dict of str to value.
        The configuration parameters.

    Example
    --------
    .. code-block:: python

      # the following code reconfigures the environment
      # temporarily to attributes specified in new_cfg.json
      new_cfg = json.load(json.load(open("new_cfg.json")))
      with vta.Environment(new_cfg):
          # env works on the new environment
          env = vta.get_env()
    """
    current = None
    # constants
    MAX_XFER = 1 << 22
    # debug flags
    DEBUG_DUMP_INSN = (1 << 1)
    DEBUG_DUMP_UOP = (1 << 2)
    DEBUG_SKIP_READ_BARRIER = (1 << 3)
    DEBUG_SKIP_WRITE_BARRIER = (1 << 4)
    # memory scopes
    inp_scope = "local.inp_buffer"
    wgt_scope = "local.wgt_buffer"
    acc_scope = "local.acc_buffer"

    # initialization function
    def __init__(self, cfg):
        self.__dict__.update(cfg)
        for key in PkgConfig.cfg_keys:
            if key not in cfg:
                raise ValueError("Expect key %s in cfg" % key)
        # derive output buffer size
        self.LOG_OUT_BUFF_SIZE = (
            self.LOG_ACC_BUFF_SIZE +
            self.LOG_OUT_WIDTH -
            self.LOG_ACC_WIDTH)
        # data type width
        self.INP_WIDTH = 1 << self.LOG_INP_WIDTH
        self.WGT_WIDTH = 1 << self.LOG_WGT_WIDTH
        self.ACC_WIDTH = 1 << self.LOG_ACC_WIDTH
        self.OUT_WIDTH = 1 << self.LOG_OUT_WIDTH
        # tensor intrinsic shape
        self.BATCH = 1 << self.LOG_BATCH
        self.BLOCK_IN = 1 << self.LOG_BLOCK_IN
        self.BLOCK_OUT = 1 << self.LOG_BLOCK_OUT
        # buffer size
        self.UOP_BUFF_SIZE = 1 << self.LOG_UOP_BUFF_SIZE
        self.INP_BUFF_SIZE = 1 << self.LOG_INP_BUFF_SIZE
        self.WGT_BUFF_SIZE = 1 << self.LOG_WGT_BUFF_SIZE
        self.ACC_BUFF_SIZE = 1 << self.LOG_ACC_BUFF_SIZE
        self.OUT_BUFF_SIZE = 1 << self.LOG_OUT_BUFF_SIZE
        # bytes per buffer
        self.INP_ELEM_BITS = (self.BATCH *
                              self.BLOCK_IN *
                              self.INP_WIDTH)
        self.WGT_ELEM_BITS = (self.BLOCK_OUT *
                              self.BLOCK_IN *
                              self.WGT_WIDTH)
        self.ACC_ELEM_BITS = (self.BATCH *
                              self.BLOCK_OUT *
                              self.ACC_WIDTH)
        self.OUT_ELEM_BITS = (self.BATCH *
                              self.BLOCK_OUT *
                              self.OUT_WIDTH)
        self.INP_ELEM_BYTES = self.INP_ELEM_BITS // 8
        self.WGT_ELEM_BYTES = self.WGT_ELEM_BITS // 8
        self.ACC_ELEM_BYTES = self.ACC_ELEM_BITS // 8
        self.OUT_ELEM_BYTES = self.OUT_ELEM_BITS // 8
        # Configuration bitstream name
        self.BITSTREAM = "{}x{}x{}_{}bx{}b_{}_{}_{}_{}_{}MHz_{}ns_v{}.bit".format(
            (1 << cfg["LOG_BATCH"]),
            (1 << cfg["LOG_BLOCK_IN"]),
            (1 << cfg["LOG_BLOCK_OUT"]),
            (1 << cfg["LOG_INP_WIDTH"]),
            (1 << cfg["LOG_WGT_WIDTH"]),
            cfg["LOG_UOP_BUFF_SIZE"],
            cfg["LOG_INP_BUFF_SIZE"],
            cfg["LOG_WGT_BUFF_SIZE"],
            cfg["LOG_ACC_BUFF_SIZE"],
            cfg["HW_FREQ"],
            cfg["HW_CLK_TARGET"],
            cfg["HW_VER"].replace('.', '_'))
        # dtypes
        self.acc_dtype = "int%d" % self.ACC_WIDTH
        self.inp_dtype = "int%d" % self.INP_WIDTH
        self.wgt_dtype = "int%d" % self.WGT_WIDTH
        self.out_dtype = "int%d" % self.OUT_WIDTH
        # lazy cached members
        self.mock_mode = False
        self._mock_env = None
        self._dev_ctx = None
        self._last_env = None

    def __enter__(self):
        self._last_env = Environment.current
        Environment.current = self
        return self

    def __exit__(self, ptype, value, trace):
        Environment.current = self._last_env

    def pkg_config(self):
        """PkgConfig instance"""
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        proj_root = os.path.abspath(os.path.join(curr_path, "../../"))
        return PkgConfig(self.__dict__, proj_root)

    @property
    def dev(self):
        """Developer context"""
        if self._dev_ctx is None:
            self._dev_ctx = DevContext(self)
        return self._dev_ctx

    @property
    def mock(self):
        """A mock version of the Environment

        The ALU, dma_copy and intrinsics will be
        mocked to be nop.
        """
        if self.mock_mode:
            return self
        if self._mock_env is None:
            self._mock_env = copy.copy(self)
            self._mock_env._dev_ctx = None
            self._mock_env.mock_mode = True
        return self._mock_env

    @property
    def dma_copy(self):
        """DMA copy pragma"""
        return ("dma_copy"
                if not self.mock_mode
                else "skip_dma_copy")

    @property
    def alu(self):
        """ALU pragma"""
        return ("alu"
                if not self.mock_mode
                else "skip_alu")

    @property
    def gemm(self):
        """GEMM intrinsic"""
        return self.dev.gemm

    @property
    def target(self):
        return tvm.target.vta(model=self.TARGET)

    @property
    def target_host(self):
        """The target host"""
        if self.TARGET == "pynq":
            return "llvm -target=armv7-none-linux-gnueabihf"
        if self.TARGET == "sim" or self.TARGET == "tsim":
            return "llvm"
        raise ValueError("Unknown target %s" % self.TARGET)

    @property
    def target_vta_cpu(self):
        return tvm.target.arm_cpu(model=self.TARGET)

def get_env():
    """Get the current VTA Environment.

    Returns
    -------
    env : Environment
        The current environment.
    """
    return Environment.current


# The memory information for the compiler
@tvm.register_func("tvm.info.mem.%s" % Environment.inp_scope)
def mem_info_inp_buffer():
    spec = get_env()
    return tvm.make.node("MemoryInfo",
                         unit_bits=spec.INP_ELEM_BITS,
                         max_simd_bits=spec.INP_ELEM_BITS,
                         max_num_bits=spec.INP_BUFF_SIZE * 8,
                         head_address=None)

@tvm.register_func("tvm.info.mem.%s" % Environment.wgt_scope)
def mem_info_wgt_buffer():
    spec = get_env()
    return tvm.make.node("MemoryInfo",
                         unit_bits=spec.WGT_ELEM_BITS,
                         max_simd_bits=spec.WGT_ELEM_BITS,
                         max_num_bits=spec.WGT_BUFF_SIZE * 8,
                         head_address=None)

@tvm.register_func("tvm.info.mem.%s" % Environment.acc_scope)
def mem_info_acc_buffer():
    spec = get_env()
    return tvm.make.node("MemoryInfo",
                         unit_bits=spec.ACC_ELEM_BITS,
                         max_simd_bits=spec.ACC_ELEM_BITS,
                         max_num_bits=spec.ACC_BUFF_SIZE * 8,
                         head_address=None)

# TVM related registration
@tvm.register_func("tvm.intrin.rule.default.vta.coproc_sync")
def coproc_sync(op):
    _ = op
    return tvm.call_extern(
        "int32", "VTASynchronize",
        get_env().dev.command_handle, 1<<31)


@tvm.register_func("tvm.intrin.rule.default.vta.coproc_dep_push")
def coproc_dep_push(op):
    return tvm.call_extern(
        "int32", "VTADepPush",
        get_env().dev.command_handle,
        op.args[0], op.args[1])


@tvm.register_func("tvm.intrin.rule.default.vta.coproc_dep_pop")
def coproc_dep_pop(op):
    return tvm.call_extern(
        "int32", "VTADepPop",
        get_env().dev.command_handle,
        op.args[0], op.args[1])


def _init_env():
    """Iniitalize the default global env"""
    curr_path = os.path.dirname(
        os.path.abspath(os.path.expanduser(__file__)))
    proj_root = os.path.abspath(os.path.join(curr_path, "../../../"))
    path_list = [
        os.path.join(curr_path, "vta_config.json"),
        os.path.join(proj_root, "build", "vta_config.json"),
        os.path.join(proj_root, "vta_config.json"),
        os.path.join(proj_root, "vta/config/vta_config.json")
    ]
    path_list = [p for p in path_list if os.path.exists(p)]
    if not path_list:
        raise RuntimeError(
            "Error: {} not found.make sure you have config.json in your vta root"
            .format(filename))
    return Environment(json.load(open(path_list[0])))

Environment.current = _init_env()
