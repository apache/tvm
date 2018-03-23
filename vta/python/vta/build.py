"""Runtime function related hooks"""
from __future__ import absolute_import as _abs

import tvm
from tvm import build_module
from . runtime import CB_HANDLE
from . import ir_pass


def lift_coproc_scope(x):
    x = ir_pass.lift_alloc_to_scope_begin(x)
    x = tvm.ir_pass.LiftAttrScope(x, "coproc_scope", False)
    return x

def early_rewrite(stmt):
    try:
        return tvm.ir_pass.StorageRewrite(stmt)
    except tvm.TVMError:
        return stmt


def debug_mode(debug_flag):
    """Pass to enable vta debug mode.

    Parameters
    ----------
    debug_flag : int
        The dbeug flag to be passed.

    Returns
    -------
    pass_list: list of function
        The pass to set to build_config(add_lower_pass=vta.debug_mode(mode))
    """
    def add_debug(stmt):
        debug = tvm.call_extern(
            "int32", "VTASetDebugMode", CB_HANDLE, debug_flag)
        return tvm.make.stmt_seq(debug, stmt)
    pass_list = [(1, ir_pass.inject_dma_intrin),
                 (1, ir_pass.inject_skip_copy),
                 (1, ir_pass.annotate_alu_coproc_scope),
                 (1, lambda x: tvm.ir_pass.LiftAttrScope(x, "coproc_uop_scope", True)),
                 (1, lift_coproc_scope),
                 (1, ir_pass.inject_coproc_sync),
                 (1, early_rewrite)]
    if debug_flag:
        pass_list.append((1, add_debug))
    pass_list.append((2, ir_pass.inject_alu_intrin))
    pass_list.append((3, ir_pass.fold_uop_loop))
    pass_list.append((3, ir_pass.cpu_access_rewrite))
    return pass_list


# Add a lower pass to sync uop
build_module.BuildConfig.current.add_lower_pass = debug_mode(0)
