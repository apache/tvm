from typing import *

import numpy as np
import tvm
from numpy.lib.type_check import imag
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.testing import densenet, mobilenet, resnet, resnet_3d, squeezenet
from tvm.relay.transform import InferType
from tvm.relay.transform.fp16_conversion import fp32_to_fp16


def run_module(mod, mod_params):
    dev = tvm.device("llvm", 0)
    intrp = relay.create_executor("debug", mod, device=dev, target="llvm")
    # in_data = [tvm.nd.array(value) for value in in_data.values()]
    return intrp.evaluate()(**mod_params).asnumpy()


def verify_fp32_fp16_output_close(mod, mod_params):
    result_fp32 = run_module(mod, mod_params)

    fp16 = fp32_to_fp16.quantize_to_fp16(mod["main"].body)
    fp16_mod = tvm.ir.IRModule.from_expr(fp16)
    result_fp16 = run_module(fp16_mod, mod_params)
    
    # Ensure the results are close
    np.testing.assert_allclose(result_fp32, result_fp16, rtol=1e-3)

def test_resnet18():
    np.random.seed(4321)
    mod, mod_params = resnet.get_workload(1, 5, num_layers=18, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_resnet18_3d():
    np.random.seed(3215)
    mod, mod_params = resnet_3d.get_workload(1, 5, num_layers=18, image_shape=(1, 3, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 3, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_mobilenet():
    np.random.seed(4615)

    mod, mod_params = mobilenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_densenet():
    np.random.seed(3222)
    mod, mod_params = densenet.get_workload(classes=5, batch_size=1, image_shape=(1, 224, 224))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 224, 224)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)


def test_squeezenet():
    np.random.seed(5628)
    mod, mod_params = squeezenet.get_workload(1, 5, image_shape=(1, 32, 32))
    mod_params["data"] = np.random.uniform(-10, 10, (1, 1, 32, 32)).astype("float32")

    verify_fp32_fp16_output_close(mod, mod_params)

#def test_
