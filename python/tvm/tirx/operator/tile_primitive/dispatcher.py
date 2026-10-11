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
"""Validation helpers for explicit tensor instructions."""


class DispatchFail(RuntimeError):
    """The selected instruction cannot implement its operands."""


def fail(reason):
    raise DispatchFail(reason)


def run_dispatch(call, sctx):
    from tvm.tirx.tensor_instruction import TensorCall

    view = TensorCall.decode(call)
    view.call.validate()
    if sctx.target.kind.name != view.spec.backend:
        raise ValueError(
            f"{view.op.name}: target {sctx.target.kind.name} does not match {view.spec.backend}"
        )
    if view.spec.backend == "trn" and sctx.scope_kind != "thread":
        raise ValueError(f"{view.op.name}: Trainium tensor instructions require thread scope")
    try:
        result = view.spec.lower(view, sctx)
        if result is None:
            raise DispatchFail("lowerer returned no implementation")
        return result
    except Exception as error:
        raise RuntimeError(
            f"Tensor instruction lowering failed: op={view.op.name} "
            f"target={sctx.target.kind.name} scope={sctx.scope_kind}\n"
            f"{error}\ncall: {view.call}"
        ) from error
