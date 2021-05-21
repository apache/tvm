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
"""Test for argwhere operator"""
import numpy as np

import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing

_argwhere_schedule = {
    "generic": topi.generic.schedule_argwhere,
    "gpu": topi.cuda.schedule_argwhere,
}

_argwhere_compute = {"llvm": topi.argwhere, "cuda": topi.cuda.argwhere}


def verify_argwhere(data_shape):
    dtype = "int32"
    np_data = np.random.choice([0, 1, 2, 3], size=data_shape).astype(dtype)
    np_out = np.argwhere(np_data)
    out_shape = np_out.shape[0]
    np_shape = np.ones(shape=(out_shape, len(data_shape)), dtype=dtype)

    out_shape = te.placeholder(shape=(out_shape, len(data_shape)), name="out_shape", dtype=dtype)
    condition = te.placeholder(shape=data_shape, name="condition", dtype=dtype)

    def check_device(target):
        dev = tvm.device(target, 0)
        if not dev.exist or target not in _argwhere_compute:
            return

        with tvm.target.Target(target):
            out = _argwhere_compute[target](out_shape, condition)
            s_func = tvm.topi.testing.dispatch(target, _argwhere_schedule)
            sch = s_func(out)

        func = tvm.build(sch, [out_shape, condition, out], target, name="argwhere")

        args = [tvm.nd.array(np_shape, dev)]
        args.append(tvm.nd.array(np_data, dev))
        args.append(tvm.nd.empty(out.shape, device=dev, dtype=condition.dtype))
        func(*args)
        np.set_printoptions(threshold=np.inf)
        tvm.testing.assert_allclose(args[-1].numpy(), np.array(np_out))

    for target, _ in tvm.testing.enabled_targets():
        check_device(target)


@tvm.testing.uses_gpu
def test_argwhere():
    verify_argwhere((1,))
    verify_argwhere((100,))
    verify_argwhere((1, 1))
    verify_argwhere((5, 3))
    verify_argwhere((32, 64))
    verify_argwhere((128, 65))
    verify_argwhere((200, 500))
    verify_argwhere((6, 5, 3))
    verify_argwhere((1, 1, 1))
    verify_argwhere((1, 1, 1, 1))
    verify_argwhere((6, 4, 5, 3))
    verify_argwhere((1, 1, 1, 1, 1))
    verify_argwhere((6, 4, 5, 3, 7))


if __name__ == "__main__":
    test_argwhere()
