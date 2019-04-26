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
""" Support level6 operator test cases.
"""
import math
import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import ctx_list
import topi.testing

def test_argsort():
    def verify_argsort(shape, axis, is_ascend):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.argsort(x, axis=axis, is_ascend=is_ascend)
        zz = relay.ir_pass.infer_type(z)
        func = relay.Function([x], z)
        x_data = np.random.uniform(size=shape).astype("float32")
        if is_ascend:
            ref_res = np.argsort(x_data, axis=axis)
        else:
            ref_res = np.argsort(-x_data, axis=axis)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.astype("float"), rtol=1e-5)
    verify_argsort((2, 3, 4), axis=0, is_ascend=False)
    verify_argsort((1, 4, 6), axis=1, is_ascend=True)
    verify_argsort((3, 5, 6), axis=-1, is_ascend=False)


if __name__ == "__main__":
    test_argsort()
