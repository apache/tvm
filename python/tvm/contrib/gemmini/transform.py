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
# pylint: disable=len-as-condition, no-else-return, unused-argument, invalid-name
"""
Transformation passes for Gemmini
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

from numpy import isin
import tvm
from tvm import te
from tvm.topi import utils
import numpy as np
from copy import deepcopy
import itertools
import ast
from tvm.tir.ir_builder import IRBuilder
from typing import Dict

from .environment import Environment

env = Environment.instance()


def _get_counters(irb: IRBuilder):
    """Generates calls to print the values of the configured timers

    Args:
        irb (IRBuilder): IRBuilder
    """
    irb.emit(tvm.tir.call_extern("", "counter_snapshot_take"))
    irb.emit(tvm.tir.call_extern("", "printf", "Counter values:\\r\\n"))
    counter_vars = []
    for i, (key, value) in enumerate(env.enabled_counters.items()):
        counter_var = irb.let(
            value.lower() + "_var", tvm.tir.call_extern("uint32", "counter_read", i)
        )
        counter_vars.append(counter_var)
        irb.emit(tvm.tir.call_extern("", "printf", tvm.tir.StringImm("%s," % value)))
    irb.emit(tvm.tir.call_extern("", "printf", "\\r\\n"))
    for c in counter_vars:
        irb.emit(tvm.tir.call_extern("", "printf", tvm.tir.StringImm("%lu,"), c))
    irb.emit(tvm.tir.call_extern("", "printf", "\\r\\n"))


def _configure_timers(irb: IRBuilder):
    """Generates calls to configure the enabled counters

    Args:
        irb (IRBuilder): IRBuilder
    """
    for i, (key, value) in enumerate(env.enabled_counters.items()):
        irb.emit(tvm.tir.call_extern("", "counter_configure", i, key))


def _reset_counters(irb: IRBuilder):
    """Generates calls to reset all Gemmini counters

    Args:
        irb (IRBuilder): IRBuilder
    """
    irb.emit(tvm.tir.call_extern("", "counter_reset"))
    irb.emit(tvm.tir.call_extern("", "counter_snapshot_reset"))


def _match_pragma(stmt, key):
    """Internal helper to match stmt to pragma stmt.

    Parameters
    ----------
    stmt : Stmt
        The AttrStmt

    key : str
        The pragma key
    """
    return (stmt.attr_key == "pragma_" + key) or (
        stmt.attr_key == "pragma_scope" and stmt.value.value == key
    )


def _get_config_dict_from_str(str_value: str) -> Dict:
    """Returns a configuration dictionary from its string representation

    Args:
        str_value (str): Dictionary encoded in a string

    Returns:
        Dict: Configuration dictionary
    """
    return ast.literal_eval(str(str_value).replace("'", '"').replace('"{', "{").replace('}"', "}"))


def _gen_debug_header(irb: IRBuilder):
    """If the debug flag is activated in the environment, generate the debug headers for the code

    Args:
        irb (IRBuilder): _description_
    """
    if env.debug:
        _configure_timers(irb)
        _reset_counters(irb)


def _gen_debug_tail(irb: IRBuilder):
    """If the debug flag is activated in the environment, generate the debug tails for the code

    Args:
        irb (IRBuilder): _description_
    """
    if env.debug:
        _get_counters(irb)


def InsertGemminiHeaderOperators():
    """Pass to generate the calls to the Gemmini configuration instructions"""

    def _do_fold(stmt):
        if _match_pragma(stmt, "add_start"):
            irb = tvm.tir.ir_builder.create()
            _gen_debug_header(irb)

            irb.emit(tvm.tir.call_extern("", "gemmini_flush", 0))

            config_dict = _get_config_dict_from_str(stmt.body.value)
            A_size = config_dict["A_size"]
            B_size = config_dict["B_size"]
            C_size = config_dict["C_size"]
            A_private_stride = config_dict["A_private_stride"]
            B_private_stride = config_dict["B_private_stride"]
            execution_stride = config_dict["execution_stride"]
            activation = config_dict["activation"]
            mode = config_dict["mode"]
            max_pixels_per_row = config_dict["max_pixels_per_row"]
            ifm1_scale = config_dict["ifm1_scale"]
            ifm2_scale = config_dict["ifm2_scale"]
            scale = config_dict["scale"]
            act = 1 if activation else 0

            shrunk = 1
            irb.emit(tvm.tir.call_extern("", "gemmini_config_ex", mode, act, 0))
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended4_config_ld",
                    A_size,
                    ifm1_scale,
                    shrunk,
                    A_private_stride,
                    0,
                )
            )
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended4_config_ld",
                    B_size,
                    ifm2_scale,
                    shrunk,
                    B_private_stride,
                    1,
                )
            )
            irb.emit(
                tvm.tir.call_extern(
                    "", "gemmini_extended4_config_ld", C_size * 4, scale, 0, env.DIM, 2
                )
            )
            irb.emit(tvm.tir.call_extern("", "gemmini_extended_config_st", C_size, act, scale))

            return tvm.tir.SeqStmt([irb.get(), stmt])
        elif _match_pragma(stmt, "gemm_start"):
            irb = tvm.tir.ir_builder.create()
            _gen_debug_header(irb)

            irb.emit(tvm.tir.call_extern("", "gemmini_flush", 0))

            config_dict = _get_config_dict_from_str(stmt.body.value)
            A_size = config_dict["A_size"]
            B_size = config_dict["B_size"]
            C_size = config_dict["C_size"]
            A_private_stride = config_dict["A_private_stride"]
            B_private_stride = config_dict["B_private_stride"]
            execution_stride = config_dict["execution_stride"]
            activation = config_dict["activation"]
            mode = config_dict["mode"]
            max_pixels_per_row = config_dict["max_pixels_per_row"]
            scale = config_dict["scale"]
            padding_value = config_dict["padding_value"]
            act = 1 if activation else 0

            irb.emit(
                tvm.tir.call_extern(
                    "", "gemmini_extended_config_ex", mode, act, 0, execution_stride, 0, 0
                )
            )
            if padding_value == 0:
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "gemmini_extended5_config_ld",
                        A_size,
                        1.0,
                        0,
                        A_private_stride,
                        max_pixels_per_row,
                        0,
                    )
                )
            else:
                irb.emit(
                    tvm.tir.call_extern(
                        "",
                        "gemmini_extended6_config_ld",
                        A_size,
                        1.0,
                        0,
                        A_private_stride,
                        max_pixels_per_row,
                        padding_value,
                        0,
                    )
                )
            irb.emit(
                tvm.tir.call_extern(
                    "", "gemmini_extended5_config_ld", B_size, 1.0, 0, B_private_stride, 1, 1
                )
            )
            irb.emit(tvm.tir.call_extern("", "gemmini_extended4_config_ld", 0, 1.0, 0, env.DIM, 2))
            irb.emit(tvm.tir.call_extern("", "gemmini_extended_config_st", C_size, act, scale))

            return tvm.tir.SeqStmt([irb.get(), stmt])
        elif _match_pragma(stmt, "gemm_cisc_start"):
            irb = tvm.tir.ir_builder.create()
            _gen_debug_header(irb)

            irb.emit(tvm.tir.call_extern("", "gemmini_flush", 0))
            return tvm.tir.SeqStmt([irb.get(), stmt])
        elif _match_pragma(stmt, "conv2d_cisc_start") or _match_pragma(
            stmt, "dw_conv2d_cisc_start"
        ):
            irb = tvm.tir.ir_builder.create()
            _gen_debug_header(irb)

            return tvm.tir.SeqStmt([irb.get(), stmt])
        return None

    def _ftransform(f, mod, ctx):
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(f.body, _do_fold, None, ["tir.AttrStmt"])
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.gemmini.insert_header_operators"
    )


def InsertGemminiFenceOperator():
    """Pass to generate the call to the fence instruction at the end of the operator"""

    func_name = ""

    def _do_fold(stmt):
        if _match_pragma(stmt, "gemm_end"):
            irb = tvm.tir.ir_builder.create()
            irb.emit(tvm.tir.call_extern("", "gemmini_fence"))
            _gen_debug_tail(irb)

            return tvm.tir.SeqStmt([stmt, irb.get()])
        return None

    def _ftransform(f, mod, ctx):
        func_name = f.attrs["global_symbol"]
        return f.with_body(
            tvm.tir.stmt_functor.ir_transform(f.body, _do_fold, None, ["tir.AttrStmt"])
        )

    return tvm.tir.transform.prim_func_pass(
        _ftransform, opt_level=0, name="tir.gemmini.insert_fence_operators"
    )


def InjectAMVINIntrin():
    """Pass to inject A mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("A mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                cols = 1
            else:
                cols = src.shape[1]
            rows = src.shape[0]
            dst_access_ptr = dst.access_ptr("w", "uint32")

            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.INP_SCR_BASE_ADDRESS, "uint8") + dst_access_ptr,
                    cols,
                    rows,
                )
            )

            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.A_mvin, _inject_copy)


def InjectAMVINIntrinTransposed():
    """Pass to inject A mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("A mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            # TODO (FP): check this pointers types again!
            if len(src.shape) == 1:
                rows = 1
            else:
                rows = src.shape[1]
            cols = src.shape[0]
            dst_access_ptr = dst.access_ptr("w", "uint32")
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.INP_SCR_BASE_ADDRESS, "uint8") + dst_access_ptr,
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.A_mvin + "_t", _inject_copy)


def InjectBMVINIntrin():
    """Pass to inject B mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        wgt_base_address = tvm.runtime.const(env.WGT_SCR_BASE_ADDRESS, "int32")
        if dst.scope() == "global":
            raise RuntimeError("B mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                cols = 1
            else:
                cols = src.shape[1]
            rows = src.shape[0]
            dst_access_ptr = dst.access_ptr("r", "int32")
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin2",
                    src.access_ptr("r"),
                    wgt_base_address + dst_access_ptr,
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.B_mvin, _inject_copy)


def InjectBMVINIntrinTransposed():
    """Pass to inject B mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("B mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                rows = 1
            else:
                rows = src.shape[1]
            cols = src.shape[0]
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin2",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.WGT_SCR_BASE_ADDRESS, "uint32")
                    + dst.access_ptr("w", "uint32"),
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.B_mvin + "_t", _inject_copy)


def InjectDMVINIntrin():
    """Pass to inject D mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("D mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                cols = 1
            else:
                cols = src.shape[1]
            rows = src.shape[0]
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin3",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + dst.access_ptr("w", "uint32")
                    - tvm.runtime.const(0x40000000, "uint32"),
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.D_mvin, _inject_copy)


def InjectDMVINIntrinTransposed():
    """Pass to inject D mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("D mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                rows = 1
            else:
                rows = src.shape[1]
            cols = src.shape[0]
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin3",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + dst.access_ptr("w", "uint32")
                    - tvm.runtime.const(0x40000000, "uint32"),
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.D_mvin + "_t", _inject_copy)


def InjectCMVOUTIntrin():
    """Pass to inject C mvout intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if src.scope() == "global":
            raise RuntimeError("C mvout should have a local source")
        elif dst.scope() == "global":
            # Store
            irb = tvm.tir.ir_builder.create()
            if len(dst.shape) == 1:
                cols = 1
            else:
                cols = dst.shape[1]
            rows = dst.shape[0]
            out_access_ptr = src.access_ptr("w", "uint32")
            get_full_width = tvm.runtime.const(0x00000000, "uint32")
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvout",
                    dst.access_ptr("w"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + out_access_ptr
                    - tvm.runtime.const(0x40000000, "uint32")
                    + get_full_width,
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.C_mvout, _inject_copy)


def InjectCMVOUTIntrinTransposed():
    """Pass to inject C mvout intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if src.scope() == "global":
            raise RuntimeError("C mvout should have a local source")
        elif dst.scope() == "global":
            # Store
            irb = tvm.tir.ir_builder.create()
            # TODO (FP): check this pointers types again!
            if len(dst.shape) == 1:
                rows = 1
            else:
                rows = dst.shape[1]
            cols = dst.shape[0]
            out_access_ptr = src.access_ptr("w", "uint32")
            get_full_width = tvm.runtime.const(0x00000000, "uint32")
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvout",
                    dst.access_ptr("w"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + out_access_ptr
                    - tvm.runtime.const(0x40000000, "uint32")
                    + get_full_width,
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.C_mvout + "_t", _inject_copy)


def InjectCMVINIntrin():
    """Pass to inject C mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("C mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                cols = 1
            else:
                cols = src.shape[1]
            rows = src.shape[0]
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + dst.access_ptr("w", "uint32")
                    - tvm.runtime.const(0x40000000, "uint32"),
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.C_mvin, _inject_copy)


def InjectCMVINIntrinTransposed():
    """Pass to inject C mvin intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("C mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                rows = 1
            else:
                rows = src.shape[1]
            cols = src.shape[0]
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + dst.access_ptr("w", "uint32")
                    - tvm.runtime.const(0x40000000, "uint32"),
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.C_mvin + "_t", _inject_copy)


def InjectCMVINAccumIntrin():
    """Pass to inject C mvin accum intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("C mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                cols = 1
            else:
                cols = src.shape[1]
            rows = src.shape[0]
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin3",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + dst.access_ptr("w", "uint32"),
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.C_mvin_accum, _inject_copy)


def InjectCMVINAccumIntrinTransposed():
    """Pass to inject C mvin accum intrinsics.

    Returns
    -------
    fpass : tvm.transform.Pass
        The pass
    """
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def _inject_copy(src, dst, pad_before, pad_after, pad_value):
        # TODO (FP): add padding support...
        _ = pad_value
        if dst.scope() == "global":
            raise RuntimeError("C mvin should have a local destination")
        elif src.scope() == "global":
            # Load
            irb = tvm.tir.ir_builder.create()
            if len(src.shape) == 1:
                rows = 1
            else:
                rows = src.shape[1]
            cols = src.shape[0]
            irb.emit(
                tvm.tir.call_extern(
                    "",
                    "gemmini_extended_mvin3",
                    src.access_ptr("r"),
                    tvm.runtime.const(env.OUT_ACC_BASE_ADDRESS, "uint32")
                    + dst.access_ptr("w", "uint32"),
                    cols,
                    rows,
                )
            )
            return irb.get()
        else:
            raise RuntimeError("Do not support copy %s->%s" % (src.scope(), dst.scope()))

    return tvm.tir.transform.InjectCopyIntrin(env.C_mvin_accum + "_t", _inject_copy)
