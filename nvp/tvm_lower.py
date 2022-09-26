import tvm
from tvm import relay
from tvm.relay.testing import init

def get_lowered_tir(mod):
    """
      Argument
      ========
      mod : IRModule that contains RelayOp only 
      
      Returns
      ========
      ret : IRModule that contains TIR only
    """
    LowerTE = tvm._ffi.get_global_func("relay.tec.LowerTE")
    host_target = tvm.target.Target("llvm")
    prim_target = tvm.target.Target("lwc", host=host_target) # FIXME: add new target device
    ctxt = tvm.transform.PassContext()
    config = tvm.target.make_compilation_config(ctxt, prim_target)
    mod = tvm.relay.transform.PlanDevices(config)(mod)
    mod = tvm.relay.transform.FuseOps(fuse_opt_level=0)(mod)
    mod = tvm.relay.transform.InferType()(mod)
    mod = LowerTE("default", config)(mod)

    keys = [key for key in mod.functions.keys()]
    return mod.functions[keys[1]]
