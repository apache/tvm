import tvm
from tvm import relay
from tvm.ir.transform import Sequential
from tvm.relay import transform
from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table
import numpy as np


@tvm.ir.register_op_attr("nn.dense", "target.libxsmm")
def dense(expr):
    args = expr.args
    print("args[0].shape", args[0].checked_type.shape)
    print("args[1].shape", args[1].checked_type.shape)
    m = int(args[0].checked_type.shape[0])
    n = int(args[1].checked_type.shape[0])
    k = int(args[0].checked_type.shape[1])
    return bool(np.cbrt(m * n * k) <= 256)

def _register_external_op_helper(op_name, supported=True):

  @tvm.ir.register_op_attr(op_name, "target.libxsmm")
  def _func_wrapper(expr):
    return supported

  return _func_wrapper

#_register_external_op_helper("nn.bias_add")
#_register_external_op_helper("nn.relu")

def get_root_call(call, root_op_name):
  if not isinstance(call, relay.Call):
    return None
  if str(call.op) == root_op_name:
    return call
  return get_root_call(call.args[0], root_op_name)

def check_dense_shape(call):
  dense = get_root_call(call, "nn.dense") 
  data = dense.args[0].checked_type
  weight = dense.args[1].checked_type
  m = int(data.shape[0])
  n = int(weight.shape[0])
  k = int(data.shape[1])
  print("m:{}, n:{}, k:{}".format(m, n, k))
  print("(m * n * k) ** (1/3): {}".format((m * n * k) ** (1/3) <= 256))
  return  (m * n * k) ** (1/3) <= 256


def make_dense_pattern(with_bias=False, eltwise=None):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    pattern_name = "libxsmm.dense"
    if with_bias:
        dense_out = is_op("nn.bias_add")(dense, bias)
        pattern_name += "_bias"
    else:
        dense_out = dense
    if eltwise:
        dense_out = is_op(eltwise)(dense_out)
        pattern_name += ("_" + eltwise.split(".")[-1])
    return [pattern_name, dense_out, check_dense_shape]


@register_pattern_table("libxsmm")
def pattern_table():
    elt_list = ["nn.relu", "sigmoid", None]
    libxsmm_patterns = []
    for with_bias in [True, False]:
        for elt in elt_list:
            libxsmm_patterns.append(make_dense_pattern(with_bias, elt))
    return libxsmm_patterns


def partition_for_libxsmm(mod, params=None):
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget(["libxsmm"]),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    return mod;
