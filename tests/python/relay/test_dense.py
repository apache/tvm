import tvm
from tvm import relay
from tvm.testing import assert_allclose
import numpy as np
from tvm.ir.instrument import pass_instrument


def _test_accuracy(input_values, output_values, build_mod):

    dev = tvm.cpu(0)

    input_buf = tvm.nd.array(input_values, device=dev)
    rt = tvm.contrib.graph_executor.GraphModule(build_mod["default"](dev))
    rt.set_input("data", input_buf)
    rt.run()
    out = rt.get_output(0)

    tvm.testing.assert_allclose(out.numpy(), output_values)


# Define input shape and data type
data_size = (64, 64)
data_shape = data_size  # Input shape
data_type = "float32"  # Data type
weight_shape = data_size

# Create Relay input variable
d = relay.var("data", shape=data_shape, dtype=data_type)
w1 = np.ones(weight_shape, dtype=data_type)
w = relay.const(w1)

# Create Relay dense layer
y = relay.nn.dense(d, w)

# Create Relay module
mod = tvm.IRModule()

# Define a Relay function with the dense layer
mod["main"] = relay.Function([d], y)

# Compile the Relay module
target = "llvm -mtriple=aarch64-linux-gnu -device=arm_cpu -mattr=+v8.2a,+neon"  # Example target, you can change this to your desired target
lib = relay.build(mod, target=target, params=None)

in_np = np.random.uniform(size=(data_size)).astype(data_type)
out_np = np.array(np.matmul(in_np, w1.T))

target = "llvm -mtriple=aarch64-linux-gnu -mattr=+sve"
_test_accuracy(in_np, out_np, lib)
