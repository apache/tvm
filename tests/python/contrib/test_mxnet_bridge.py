# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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


    tvm.testing.assert_allclose(
        zz.asnumpy(), (xx.asnumpy() + yy.asnumpy()) * 10)


if __name__ == "__main__":
    mxnet_check()
