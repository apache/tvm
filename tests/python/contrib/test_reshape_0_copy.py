import torch
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.frontend.common import infer_shape

def test_reshape_0_copy():
    shape0 = (56, 224)
    shape1 = (112, 112)
    in_name0 = "infeats0"
    in_name1 = "infeats1"
    x0 = relay.var(in_name0, shape=shape0, dtype="float32")
    x0 = relay.reshape(x0, shape1)

    x1 = relay.var(in_name1, shape=shape1, dtype="float32")
    mat = relay.nn.matmul(x0, x1)
    _y = relay.reshape(mat, (-1))
    func = relay.Function(relay.analysis.free_vars(_y), _y)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target="llvm")
    m = graph_executor.GraphModule(lib['default'](tvm.cpu(0)))

    data0 = torch.rand(shape0, dtype=torch.float32, device=torch.device("cpu:0"))
    data1 = torch.rand(shape1, dtype=torch.float32, device=torch.device("cpu:0"))

    data_ndarray0 = tvm.nd.from_dlpack(torch.to_dlpack(data0))
    data_ndarray1 = tvm.nd.from_dlpack(torch.to_dlpack(data1))

    def expected():
        m.set_input(in_name0, data_ndarray0)
        m.set_input(in_name1, data_ndarray1)
        m.run()
        return torch.from_dlpack(m.get_output(0)).cpu().detach().numpy()

    def zero_copy():
        outshape = infer_shape(_y)
        output = torch.empty(outshape, device=torch.device("cpu:0"))
        output_view = tvm.nd.from_dlpack(torch.to_dlpack(output))
        m.set_input_zero_copy(in_name0, data_ndarray0)
        m.set_input_zero_copy(in_name1, data_ndarray1)
        m.set_output_zero_copy(0, output_view)
        m.run()
        return output.cpu().detach().numpy()

    golden_out = expected()
    out = zero_copy()
    np.testing.assert_equal(golden_out, out)

if __name__ == "__main__":
    test_reshape_0_copy()