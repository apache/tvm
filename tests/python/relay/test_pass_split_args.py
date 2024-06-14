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


target_name = tvm.testing.parameter("opencl", "metal", "cuda")
shape_type = tvm.testing.parameter("dynamic", "static")


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_split_concat(target_name, shape_type):
    if shape_type == "dynamic":
        shape = (tvm.tir.Any(), 1, 1, 3)
        number_of_any_dims = 1
    else:
        shape = (1, 1, 1, 3)
        number_of_any_dims = 0
    ndims = len(shape)
    dtype = "float32"
    axis = 1
    tensors_num = 300
    inputs = []
    for i in range(tensors_num):
        inputs.append(relay.var("p{}".format(i), shape=shape, dtype=dtype))

    def before():
        inp = relay.Tuple(inputs)
        return relay.op.concatenate(inp, axis)

    def expected(limit):
        if limit == 0:
            return before()
        limit = limit - 1  # one buffer with output
        if number_of_any_dims > 0:
            limit -= ndims

        new_args = []
        added_args = 0
        num_inputs = 0
        for inp in inputs:
            curr_args = 1 + number_of_any_dims
            if number_of_any_dims > 0:
                curr_args += ndims
            num_inputs += curr_args
            if added_args + curr_args > limit:
                t = relay.Tuple(new_args)
                stop = relay.annotation.stop_fusion(t)
                concat = relay.op.concatenate(stop, axis)
                new_args = [concat]
                added_args = curr_args
            added_args += curr_args
            new_args.append(inp)
        t = relay.Tuple(new_args)
        stop = relay.annotation.stop_fusion(t)
        concat = relay.op.concatenate(stop, axis)

        if num_inputs < limit:
            return before()

        return concat

    # the fold constant should work on any context.
    limit = tvm.target.Target(target_name).max_function_args
    res = run_opt_pass(before(), transform.SplitArgs(limit))
    exp = run_opt_pass(expected(limit), transform.InferType())
    tvm.ir.assert_structural_equal(res, exp)


if __name__ == "__main__":
    tvm.testing.main()
