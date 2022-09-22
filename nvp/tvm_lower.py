import tvm
from tvm import relay
from tvm.relay.testing import init

def gen_module(layer):
  if layer == 'relu':
    data = relay.var("data", shape=[1, 100], dtype="float32")
    Trelay = tvm.relay.nn.relu(data)
  elif layer == 'leaky_relu':
    data = relay.var("data", shape=[1, 100], dtype="float32")
    Trelay = tvm.relay.nn.leaky_relu(data)
  elif layer == 'softmax':
    data = relay.var("data", shape=[1, 100], dtype="float32")
    Trelay = tvm.relay.nn.softmax(data)
  elif layer == 'conv2d':
    data = relay.var("data", shape=[1, 10, 5, 5], dtype="float32") #NCHW
    weight = relay.var("weight", shape=[7, 10, 3, 3], dtype="float32") #OIHW
    Trelay = tvm.relay.nn.conv2d(data, weight, strides=(1,1), padding=(0,0))
  elif layer == 'dwconv_3x3':
    data = relay.var("data", shape=[1, 10, 5, 5], dtype="float32") #NCHW8c
    weight = relay.var("weight", shape=[7, 10, 3, 3], dtype="float32") #OIHW
    Trelay = tvm.relay.nn.conv2d(data, weight, strides=(1,1), padding=(0,0))
  else:
    raise NotImplementedError("Currently NOT implemented: %s" %(layer))

  # data = relay.var("data", shape=[1, 10, 5, 5], dtype="float32")
  # Trelay = tvm.relay.nn.upsampling(data, scale_h=2, scale_w=2)
  # Trelay = tvm.relay.nn.avg_pool2d(data, pool_size=(3,3))
  # Trelay = tvm.relay.nn.max_pool2d(data, pool_size=(3,3))

  args = relay.analysis.free_vars(Trelay)   # tvm.relay.Var
  mod = tvm.IRModule.from_expr(Trelay)
  func = relay.Function(args, Trelay)
  # mod = tvm.IRModule.from_expr(func)
  mod, params = init.create_workload(func)
  return mod

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
    return mod

# def @main(%data {virtual_device=VirtualDevice(device_type=11, virtual_device_id=0, target=Target(kind='lwc', keys={'x330'}, host=Target(kind='llvm', keys={'cpu'}, attrs={'link-params': (bool)0})))}: Tensor[(1, 10, 5, 5), float32] /* ty=Tensor[(1, 10, 5, 5), float32] */, virtual_device=VirtualDevice(device_type=11, virtual_device_id=0, target=Target(kind='lwc', keys={'x330'}, host=Target(kind='llvm', keys={'cpu'}, attrs={'link-params': (bool)0})))) -> Tensor[(1, 10, 10, 10), float32] {
#   %0 = (%data,) /* ty=(Tensor[(1, 10, 5, 5), float32],) */;
#   call_lowered(@default_fused_nn_upsampling, %0, metadata={"relay_attrs"={__dict__={"Primitive"=1}}, "all_prim_fn_vars"=['default_fused_nn_upsampling']}) /* ty=Tensor[(1, 10, 10, 10), float32] */
# }
