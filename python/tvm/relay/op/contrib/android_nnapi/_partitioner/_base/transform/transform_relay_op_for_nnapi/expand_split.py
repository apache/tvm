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
"""Expand Relay IR split for mapping to Android NNAPI."""
import tvm


SPLIT_OP = tvm.relay.op.get("split")


class ExpandSplit(tvm.relay.ExprMutator):
    """Expand Relay IR split for mapping to Android NNAPI."""

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

    def visit_tuple_getitem(self, op):
        if isinstance(op.tuple_value, tvm.relay.Call) and op.tuple_value.op == SPLIT_OP:
            split = op.tuple_value
            data = split.args[0]
            new_strided_slice = {}
            new_strided_slice["data"] = self.visit(data)
            if isinstance(split.attrs["indices_or_sections"], (int, tvm.tir.IntImm)):
                section_size = int(data.checked_type.shape[split.attrs["axis"]]) // int(
                    split.attrs["indices_or_sections"]
                )
                indices = [section_size * i for i in range(int(split.attrs["indices_or_sections"]))]
            else:
                indices = [0]
                indices += list(map(int, split.attrs["indices_or_sections"]))

            split_attrs_axis = (
                int(split.attrs["axis"])
                if split.attrs["axis"] >= 0
                else int(len(data.checked_type.shape) + split.attrs["axis"])
            )
            new_strided_slice["begin"] = [
                (0 if i != split_attrs_axis else indices[op.index])
                for i in range(len(data.checked_type.shape))
            ]
            new_strided_slice["end"] = [
                (
                    int(data.checked_type.shape[i])
                    if i != split_attrs_axis
                    else (
                        indices[op.index + 1]
                        if op.index < len(indices) - 1
                        else int(data.checked_type.shape[split.attrs["axis"]])
                    )
                )
                for i in range(len(data.checked_type.shape))
            ]
            return tvm.relay.strided_slice(**new_strided_slice)
        return super().visit_tuple_getitem(op)
