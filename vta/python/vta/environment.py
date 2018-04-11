"""Configurable VTA Hareware Environment scope."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs

import os
import copy

try:
    # Allow missing import in config mode.
    import tvm
    from . import intrin
except ImportError:
    pass


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
    of developer related stuffs and user facing attributes.
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
    QID_STORE_INP = 3

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
        self.gevm = intrin.gevm(env, env.mock_mode)

    def get_task_qid(self, qid):
        """Get transformed queue index."""
        return 1 if self.DEBUG_NO_SYNC else qid


class Environment(object):
    """Hareware configuration object.

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
    cfg_keys = [
        "target",
        "LOG_INP_WIDTH",
        "LOG_WGT_WIDTH",
        "LOG_ACC_WIDTH",
        "LOG_BATCH",
        "LOG_BLOCK_IN",
        "LOG_BLOCK_OUT",
        "LOG_UOP_BUFF_SIZE",
        "LOG_INP_BUFF_SIZE",
        "LOG_WGT_BUFF_SIZE",
        "LOG_ACC_BUFF_SIZE",
    ]
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
        # Log of input/activation width in bits
        self.__dict__.update(cfg)
        for key in self.cfg_keys:
            if key not in cfg:
                raise ValueError("Expect key %s in cfg" % key)
        self.LOG_OUT_WIDTH = self.LOG_INP_WIDTH
        self.LOG_OUT_BUFF_SIZE = (
            self.LOG_ACC_BUFF_SIZE +
            self.LOG_OUT_WIDTH -
            self.LOG_ACC_WIDTH)
        # width
        self.INP_WIDTH = 1 << self.LOG_INP_WIDTH
        self.WGT_WIDTH = 1 << self.LOG_WGT_WIDTH
        self.ACC_WIDTH = 1 << self.LOG_ACC_WIDTH
        self.BATCH = 1 << self.LOG_BATCH
        self.BLOCK_IN = 1 << self.LOG_BLOCK_IN
        self.BLOCK_OUT = 1 << self.LOG_BLOCK_OUT
        self.OUT_WIDTH = self.INP_WIDTH
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
                              self.BLOCK_IN *
                              self.ACC_WIDTH)
        self.INP_ELEM_BYTES = self.INP_ELEM_BITS // 8
        self.WGT_ELEM_BYTES = self.WGT_ELEM_BITS // 8
        self.ACC_ELEM_BYTES = self.ACC_ELEM_BITS // 8
        # dtypes
        self.acc_dtype = "int%d" % self.ACC_WIDTH
        self.inp_dtype = "int%d" % self.INP_WIDTH
        self.wgt_dtype = "int%d" % self.WGT_WIDTH
        # lazy cached members
        self.mock_mode = False
        self._mock_env = None
        self._dev_ctx = None

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
    def gevm(self):
        """GEMM intrinsic"""
        return self.dev.gevm


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
def mem_info_out_buffer():
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
    python_vta_dir = os.path.dirname(__file__)
    filename = os.path.join(python_vta_dir, '../../config.mk')
    keys = set()

    for k in Environment.cfg_keys:
        keys.add("VTA_" + k)

    if not os.path.isfile(filename):
        raise RuntimeError(
            "Error: {} not found.make sure you have config.mk in your vta root"
            .format(filename))

    cfg = {}
    with open(filename) as f:
        for line in f:
            for k in keys:
                if k  +" =" in line:
                    val = line.split("=")[1].strip()
                    cfg[k[4:]] = int(val)
    cfg["target"] = "pynq"
    return Environment(cfg)

Environment.current = _init_env()
