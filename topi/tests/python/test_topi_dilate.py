import tvm
import topi
import topi.testing
import numpy as np


def test_dilate():
    target = 'llvm'
    ctx = tvm.cpu(0)

    def _test_dilate(input_size, strides):
        Input = tvm.placeholder((input_size))
        Output = topi.nn.dilate(Input, strides)
        schedule = tvm.create_schedule(Output.op)
        input_np = np.random.uniform(size=input_size).astype(Input.dtype)
        output_np = topi.testing.dilate_python(input_np, strides)
        input_tvm = tvm.nd.array(input_np, ctx=ctx)
        output_size = topi.util.get_const_tuple(Output.shape)
        output_tvm = tvm.nd.array(np.zeros(shape=output_size).astype(Output.dtype), ctx=ctx)
        f = tvm.build(schedule, [Input, Output], target)
        f(input_tvm, output_tvm)
        np.testing.assert_allclose(output_tvm.asnumpy(), output_np, rtol=1e-5)

    _test_dilate((32,), (2,))
    _test_dilate((32,32), (2,2))
    _test_dilate((1,3,32,32), (1,1,1,1))
    _test_dilate((1,3,32,32), (2,2,2,2))
    _test_dilate((1,32,32,3,3), (1,1,1,1,1))
    _test_dilate((1,32,32,3,3), (2,2,2,2,2))
    _test_dilate((1,32,32,32,3,3), (1,1,1,2,2,2))
    _test_dilate((1,32,32,32,3,3), (2,2,2,1,1,1))


if __name__ == "__main__":
    test_dilate()
