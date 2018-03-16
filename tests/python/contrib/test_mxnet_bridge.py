def mxnet_check():
    """This is a simple test function for MXNet bridge

    It is not included as nosetests, because of its dependency on mxnet

    User can directly run this script to verify correctness.
    """
    import mxnet as mx
    import topi
    import tvm
    import numpy as np
    from tvm.contrib.mxnet import to_mxnet_func

    # build a TVM function through topi
    n = 20
    shape = (20,)
    scale = tvm.var("scale", dtype="float32")
    x = tvm.placeholder(shape)
    y = tvm.placeholder(shape)
    z = topi.broadcast_add(x, y)
    zz = tvm.compute(shape, lambda *i: z(*i) * scale)

    target = tvm.target.cuda()

    # build the function
    with target:
        s = topi.generic.schedule_injective(zz)
        f = tvm.build(s, [x, y, zz, scale])

    # get a mxnet version
    mxf = to_mxnet_func(f, const_loc=[0, 1])

    ctx = mx.gpu(0)
    xx = mx.nd.uniform(shape=shape, ctx=ctx)
    yy = mx.nd.uniform(shape=shape, ctx=ctx)
    zz = mx.nd.empty(shape=shape, ctx=ctx)

    # invoke myf: this runs in mxnet engine
    mxf(xx, yy, zz, 10.0)
    mxf(xx, yy, zz, 10.0)


    np.testing.assert_allclose(
        zz.asnumpy(), (xx.asnumpy() + yy.asnumpy()) * 10)


if __name__ == "__main__":
    mxnet_check()
