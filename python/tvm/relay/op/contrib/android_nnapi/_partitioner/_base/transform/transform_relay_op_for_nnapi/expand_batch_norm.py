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
"""Expand Relay IR batch_norm for mapping to Android NNAPI
"""
import tvm


NN_BATCH_NORM_OP = tvm.relay.op.get("nn.batch_norm")


class ExpandBatchNorm(tvm.relay.ExprMutator):
    """Expand Relay IR batch_norm for mapping to Android NNAPI"""

    def __call__(self, mod):
        assert isinstance(mod, tvm.IRModule)
        ret = tvm.IRModule()
        gvs = mod.get_global_vars()
        for gvar in gvs:
            func = mod[gvar]
            func = self.visit(func)
            ret[gvar] = func
        ret = tvm.relay.transform.InferType()(ret)
        return ret

    def _expand_batch_norm(self, tgi):
        batch_norm = {}

        def _scope():
            new_args = [self.visit(a) for a in tgi.tuple_value.args]
            batch_norm["args"] = dict(
                zip(["data", "gamma", "beta", "moving_mean", "moving_var"], new_args)
            )

        _scope()
        batch_norm["attrs"] = tgi.tuple_value.attrs

        assert all(
            [
                len(batch_norm["args"][f].checked_type.shape) == 1
                for f in ["gamma", "beta", "moving_mean", "moving_var"]
            ]
        )

        # reshape args
        data_type = tgi.tuple_value.args[0].checked_type
        data_rank = len(data_type.shape)
        for arg_name in ["gamma", "beta", "moving_mean", "moving_var"]:
            target_shape = (
                [1 for i in range(0, batch_norm["attrs"]["axis"])]
                + list([int(i) for i in batch_norm["args"][arg_name].checked_type.shape])
                + [1 for i in range(batch_norm["attrs"]["axis"] + 1, data_rank)]
            )
            batch_norm["args"][arg_name] = tvm.relay.reshape(
                batch_norm["args"][arg_name], target_shape
            )

        # start expanding
        step = []
        step.append(
            batch_norm["args"]["moving_var"]
            + tvm.relay.const(batch_norm["attrs"]["epsilon"], data_type.dtype)
        )
        step.append(tvm.relay.sqrt(step[0]))
        step.append(batch_norm["args"]["data"] - batch_norm["args"]["moving_mean"])
        step.append(step[2] / step[1])
        step.append(step[3] * batch_norm["args"]["gamma"])
        step.append(step[4] + batch_norm["args"]["beta"])

        return step[-1]

    def visit_tuple_getitem(self, op):
        if (
            isinstance(op.tuple_value, tvm.relay.Call)
            and op.tuple_value.op == NN_BATCH_NORM_OP
            and op.index == 0
        ):
            return self._expand_batch_norm(op)
        return super().visit_tuple_getitem(op)
