import tvm
from tvm import relay
from tvm.relay.testing import init

def get_lowered_tir(module, model):
    """
      Argument
      ========
      module : IRModule that contains RelayOp only
      
      Returns
      ========
      ret : IRModule that contains TIR only
    """
    LowerTE = tvm._ffi.get_global_func("relay.tec.LowerTE")
    if model == "llvm":
      prim_target =tvm.target.Target("llvm")
    elif model == "hexagon":
      prim_target = tvm.target.Target("hexagon")
    elif model == "cuda":
      prim_target = tvm.target.Target("cuda")
    elif model == "x220":
      prim_target = tvm.target.Target("x220")
    else:
      raise NotImplementedError("Currently NOT supported model: %s"%(model))
    ctxt = tvm.transform.PassContext()
    config = tvm.target.make_compilation_config(ctxt, prim_target)
    module = tvm.relay.transform.PlanDevices(config)(module)
    module = tvm.relay.transform.FuseOps(fuse_opt_level=0)(module)
    module = tvm.relay.transform.InferType()(module)
    module = LowerTE("default", config)(module)

    keys = [key for key in module.functions.keys()]
    return module.functions[keys[1]]
