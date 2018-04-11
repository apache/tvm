"""VTA specific buildin for runtime."""
from __future__ import absolute_import as _abs

import tvm
from . import ir_pass
from .environment import get_env


def lift_coproc_scope(x):
    """Lift coprocessings cope to the """
    x = ir_pass.lift_alloc_to_scope_begin(x)
    x = tvm.ir_pass.LiftAttrScope(x, "coproc_scope", False)
    return x

def early_rewrite(stmt):
    """Try to do storage rewrite in early pass."""
    try:
        return tvm.ir_pass.StorageRewrite(stmt)
    except tvm.TVMError:
        return stmt


def build_config(debug_flag=0, **kwargs):
    """Build a build config for VTA.

    Parameters
    ----------
    debug_flag : int
        The dbeug flag to be passed.

    kwargs : dict
        Additional configurations.

    Returns
    -------
    build_config: BuildConfig
        The build config that can be used in TVM.

    Example
    --------
    .. code-block:: python

      # build a vta module.
      with vta.build_config():
          vta_module = tvm.build(s, ...)
    """
    env = get_env()
    def add_debug(stmt):
        debug = tvm.call_extern(
            "int32", "VTASetDebugMode",
            env.dev.command_handle,
            debug_flag)

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
    return tvm.build_config(add_lower_pass=pass_list, **kwargs)


def lower(*args, **kwargs):
    """Thin wrapper of tvm.lower

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.lower : The original TVM's lower function
    """
    cfg = tvm.build_module.current_build_config()
    if not cfg.add_lower_pass:
        with build_config():
            return tvm.lower(*args, **kwargs)
    return tvm.lower(*args, **kwargs)


def build(*args, **kwargs):
    """Thin wrapper of tvm.build

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.build : The original TVM's build function
    """
    cfg = tvm.build_module.current_build_config()
    if not cfg.add_lower_pass:
        with build_config():
            return tvm.build(*args, **kwargs)
    return tvm.build(*args, **kwargs)
