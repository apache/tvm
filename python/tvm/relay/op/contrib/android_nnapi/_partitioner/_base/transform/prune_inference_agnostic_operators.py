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
"""Remove operators that does not change inference results."""
import tvm

NN_DROPOUT_OP = tvm.relay.op.get("nn.dropout")


class PruneInferenceAgnosticOperators:
    """Remove operators that does not change inference results."""

    class _OperatorPruner(tvm.relay.ExprMutator):
        def visit_tuple_getitem(self, op):
            if (
                isinstance(op.tuple_value, tvm.relay.Call)
                and op.tuple_value.op == NN_DROPOUT_OP
                and op.index == 0
            ):
                return super().visit(op.tuple_value.args[0])
            return super().visit_tuple_getitem(op)

    def __call__(self, mod):
        """Remove operators that does not change inference results.

        Parameters
        ----------
        mod: tvm.IRModule
            The module to be pruned.

        Returns
        -------
        mod: tvm.IRModule
            The pruned module.
        """
        assert isinstance(mod, tvm.IRModule)
        ret = tvm.IRModule()
        gvs = mod.get_global_vars()
        for gvar in gvs:
            func = mod[gvar]
            func = PruneInferenceAgnosticOperators._OperatorPruner().visit(func)
            ret[gvar] = func
        return ret
