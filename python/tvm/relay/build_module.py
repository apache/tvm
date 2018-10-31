from ..build_module import build as tvm_build_module
from . codegen import GraphRuntimeCodegen
from . import ir_pass

def build(env, func, target=None):
    """
    Compile a single function to the components needed by the
    TVM RTS.

    Parameters
    ----------
    func: relay.Expr
        The function to build.

    target: optional str
        The target platform.

    Returns
    -------
    (graph_json, mod, params): tuple of (str, tvm.Module, dict)
        The outputs of building a Relay function for the TVM runtime.

    """
    if target is None:
        target = 'llvm'

    comp = GraphRuntimeCodegen(env)
    # NB(@jroesch) This creates lowered functions, and generates names for them
    #
    # We need these names to emit the correct graph as these are names of the
    # functions contained in the module.
    lowered_ops = ir_pass.lower_ops(env, func)
    mod = tvm_build_module([lf.lowered_func for lf in lowered_ops], target)

    # Therefore the call to compile must come after.
    comp.codegen(func)
    graph_json = comp.to_json()
    return graph_json, mod, None  # params currently isn't supported by API
