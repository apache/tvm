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
import numpy as np
import tvm
import tvm.testing
from tvm import topi
import tvm.topi.testing


inverse = tvm.testing.parameter(False, True)
shape = tvm.testing.parameter((7,), (3, 7), (3, 4, 5))
dtype = tvm.testing.parameter("float16", "float32", "float64")


def numpy_reference(inverse, re: np.ndarray, im: np.ndarray):
    if inverse:
        reference = np.fft.ifft(re + 1j * im)
    else:
        reference = np.fft.fft(re + 1j * im)
    return np.real(reference), np.imag(reference)


def test_dft(target, dev, inverse, shape, dtype):
    Re = tvm.te.placeholder(shape, dtype=dtype, name="Re")
    Im = tvm.te.placeholder(shape, dtype=dtype, name="Im")

    with tvm.target.Target(target):
        fcompute = topi.dft
        fschedule = topi.generic.schedule_extern

        outs = fcompute(Re, Im, inverse)
        s = fschedule(outs)

        f = tvm.build(s, [Re, Im, *outs], target)

    re_np = np.random.normal(size=shape).astype(dtype)
    im_np = np.random.normal(size=shape).astype(dtype)

    re = tvm.nd.array(re_np, device=dev)
    im = tvm.nd.array(im_np, device=dev)
    re_out = tvm.nd.array(np.zeros(shape).astype(dtype), device=dev)
    im_out = tvm.nd.array(np.zeros(shape).astype(dtype), device=dev)

    f(re, im, re_out, im_out)

    re_reference, im_reference = numpy_reference(inverse, re_np, im_np)
    tvm.testing.assert_allclose(re_out.numpy(), re_reference, rtol=1e-3, atol=1e-3)
    tvm.testing.assert_allclose(im_out.numpy(), im_reference, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    tvm.testing.main()
