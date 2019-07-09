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
from tvm import relay
from tvm.relay.transform import gradient
from tvm.relay.testing import ctx_list


def run_infer_type(expr):
    mod = relay.Module.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    return mod["main"]

def test_clip():
    ref = (lambda x: np.where(x > 10.0, np.zeros_like(x),
                     np.where(x < 1.0, np.zeros_like(x), np.ones_like(x))))
    x = relay.var("x", relay.TensorType((10, 4), "float32"))
    y = tvm.relay.clip(x, 1.0, 10.0)

    data = np.random.rand(10, 4).astype("float32") * 11.0
    ref_grad = ref(data)
    fwd_func = relay.Function([x], y)
    bwd_func = run_infer_type(gradient(fwd_func))

    for target, ctx in ctx_list():
        intrp = relay.create_executor(ctx=ctx, target=target)
        op_res, (op_grad, ) = intrp.evaluate(bwd_func)(data)
        np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)


if __name__ == "__main__":
    test_clip()
