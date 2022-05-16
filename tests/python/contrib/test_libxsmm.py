import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay.op.contrib import libxsmm
from tvm.relay.op.contrib.libxsmm import partition_for_libxsmm
from tvm.contrib import graph_runtime
from tvm import testing


def run_and_verify(mod, input_dict, params, target):
    dev = tvm.cpu()

    #print("input_dict:", input_dict)
    
    result_dict = {}
    for r in ("origin", "libxsmm"):
    #for r in ("origin", ):
    #for r in ("libxsmm", ):
      if r == "libxsmm":
        mod = partition_for_libxsmm(mod)

      print("mod:", mod)

      json, lib, param = relay.build(mod, target="llvm", params=params)
      runtime_module = tvm.contrib.graph_runtime.create(json, lib, device=dev)
      for k, v in input_dict.items():
        runtime_module.set_input(k, v)
      runtime_module.load_params(tvm.runtime.save_param_dict(param))
      runtime_module.run()
      result_dict[r] = runtime_module.get_output(0).asnumpy()
      #print("{}: {}".format(r,result_dict[r]))

    tvm.testing.assert_allclose(result_dict["origin"], result_dict["libxsmm"], rtol=1e-5, atol=1e-5)
    

def run_and_verify_func(config, dtype):
    f, input_shapes, param_list = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype) for x in param_list}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype)
        #k : np.arange(np.prod(v)).reshape(v)
        for k, v in input_shapes.items()
        if k not in param_list
    }
    #input_dict["kernel"] = np.arange(8, 8 + np.prod(input_shapes["kernel"])).reshape(input_shapes["kernel"])
    run_and_verify(f, input_dict, params, target="llvm")

def get_dense(x_shape, k_shape, dtype="float32"):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.dense(x, kernel, units=k_shape[0])
    dic = {"x": x_shape, "kernel": k_shape}
    param_list = ["kernel"]
    #param_list = []
    return out, dic, param_list


def get_dense_with_bias(x_shape, k_shape, dtype="float32"):
    dense, dic, param_list = get_dense(x_shape, k_shape, dtype)
    bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
    out = relay.nn.bias_add(dense, bias)
    dic["bias"] = (k_shape[0], )
    param_list += ["bias"]

    return out, dic, param_list


def test_dense():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)
    #x_shape = (2, 2)
    #k_shape = (2, 2)
    #x_shape = (1024, 2048)
    #k_shape = (4096, 2048)

    dense, dic, param_list = get_dense(x_shape, k_shape, dtype=dtype)
    dense = tvm.IRModule.from_expr(dense)
    config = dense, dic, param_list
    run_and_verify_func(config, dtype=dtype)
 

def test_dense_with_bias():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)
    #x_shape = (1024, 1024)
    #k_shape = (1024, 1024)

    dense_with_bias, dic, param_list = get_dense_with_bias(x_shape, k_shape, dtype=dtype)
    dense_with_bias = tvm.IRModule.from_expr(dense_with_bias)
    config = dense_with_bias, dic, param_list
    run_and_verify_func(config, dtype=dtype)


def test_dense_with_relu():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)
    #x_shape = (1024, 2048)
    #k_shape = (4096, 2048)
    dense, dic, param_list = get_dense(x_shape, k_shape, dtype=dtype)
    dense_with_relu = relay.nn.relu(dense)
    dense_with_relu = tvm.IRModule.from_expr(dense_with_relu)
    config = dense_with_relu, dic, param_list
    run_and_verify_func(config, dtype=dtype)

def test_dense_with_bias_and_relu():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)
    #x_shape = (1024, 1024)
    #k_shape = (1024, 1024)

    dense_with_bias, dic, param_list = get_dense_with_bias(x_shape, k_shape, dtype=dtype)
    dense_with_bias_and_relu = relay.nn.relu(dense_with_bias)
    dense_with_bias_and_relu = tvm.IRModule.from_expr(dense_with_bias_and_relu)
    config = dense_with_bias_and_relu, dic, param_list
    run_and_verify_func(config, dtype=dtype)

if __name__ == "__main__":
    pytest.main([__file__])
    
    #test_dense()
    #test_dense_with_bias()
    #test_dense_with_relu()
    #test_dense_with_bias_and_relu()

