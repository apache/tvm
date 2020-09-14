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
import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
import tvm.topi.testing


# TODO(mbrookhart): Enable when VM supports heterogenus execution
# @tvm.testing.uses_gpu
def test_dynamic_strided_slice():
    def verify(dshape, begin, end, strides, output, slice_mode="end", test_ref=True, dtype="int32"):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        ndim = len(dshape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(dshape)
        if strides:
            if len(strides) == 1:
                strides = strides * ndim
        else:
            strides = [1] * ndim

        # target numpy result
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.strided_slice_python(x_data, begin, end, strides, slice_mode)
        data = [x_data, np.array(begin), np.array(end)]

        begin = relay.const(begin, dtype=dtype)
        end = relay.const(end, dtype=dtype)

        if strides:
            data.append(np.array(strides))
            strides = relay.const(strides, dtype=dtype)
            z = relay.strided_slice(x, begin=begin, end=end, strides=strides, slice_mode=slice_mode)
        else:
            z = relay.strided_slice(x, begin=begin, end=end, slice_mode=slice_mode)
        func = relay.Function([x], z)

        func = run_infer_type(func)
        text = func.astext()

        if not test_ref:
            return
        for target, ctx in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor("vm", mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate()(x_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    verify((1, 3, 10, 10), [0, 0, 0, 0], [-1, 3, 10, 10], [1], (0, 3, 10, 10), dtype="int64")
    verify(
        (1, 224, 224, 3),
        [0, 20, 20, 0],
        [1, 140, 140, 3],
        [1, 1, 1, 1],
        (1, 120, 120, 3),
        dtype="int64",
    )
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], (1, 3, 3), dtype="int16")
    verify((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], (3, 1, 2))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 4], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], (1, 4, 3))
    verify((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], (1, 2, 3))
    verify(
        (3, 4, 3), [1, 0, 0], [3, -1, 3], [1, 1, 1], (2, 4, 3), slice_mode="size", test_ref=False
    )
    verify((3, 4, 3), [1, 0, 0], [-1, 2, 3], [1, 1, 1], (2, 2, 3), slice_mode="size", test_ref=True)


if __name__ == "__main__":
    test_dynamic_strided_slice()
