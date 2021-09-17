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
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_split_concat_metal():
    shape = (1, 1, 1, 3)
    dtype = "float32"
    axis = 1
    inputs = []
    for i in range(100):
        inputs.append(relay.var("p{}".format(i), shape=shape, dtype=dtype))

    def before():
        inp = relay.Tuple(inputs)
        return relay.op.concatenate(inp, axis)

    def expected():
        limit = tvm.target.Target("metal").max_function_args - 1  # one buffer with output
        splitNum = int(len(inputs) / limit)
        if len(inputs) % limit > 0:
            splitNum += 1

        splitted = []
        for i in range(splitNum):
            startIdx = i * limit
            argsCount = min(limit, len(inputs) - startIdx)
            args = []
            for j in range(argsCount):
                args.append(inputs[j + startIdx])
            t = relay.Tuple(args)
            concat = relay.op.concatenate(t, axis)
            splitted.append(relay.annotation.stop_fusion(concat))
        inp = relay.Tuple(splitted)
        return relay.op.concatenate(inp, axis)

    # the fold constant should work on any context.
    res = run_opt_pass(before(), transform.SplitArgs(tvm.target.Target("metal").max_function_args))
    exp = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(res, exp)


def test_split_concat_cuda():
    shape = (1, 1, 1, 3)
    dtype = "float32"
    axis = 1
    inputs = []
    for i in range(100):
        inputs.append(relay.var("p{}".format(i), shape=shape, dtype=dtype))

    def before():
        inp = relay.Tuple(inputs)
        return relay.op.concatenate(inp, axis)

    def expected():
        inp = relay.Tuple(inputs)
        return relay.op.concatenate(inp, axis)

    # the fold constant should work on any context.
    res = run_opt_pass(before(), transform.SplitArgs(tvm.target.Target("cuda").max_function_args))
    exp = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(res, exp)


if __name__ == "__main__":
    test_split_concat_metal()
    test_split_concat_cuda()
