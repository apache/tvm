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
import tvm.testing
from tvm import s_tir, tirx


def test_reused_block_iterator():
    """A shared block defines a fresh iterator at each realization."""
    var = tirx.Var("v", "int32")
    iterator = tirx.IterVar(tvm.ir.Range(0, 4), var, tirx.IterVar.DataPar)
    block = s_tir.SBlock([iterator], [], [], "block", tirx.Evaluate(var))
    realize = s_tir.SBlockRealize([0], True, block)
    before = tirx.PrimFunc([], tirx.SeqStmt([realize, realize]))

    after = s_tir.transform.ConvertSSA()(tvm.IRModule.from_expr(before))["main"]

    first, second = [realize.block for realize in after.body.seq]
    assert not first.iter_vars[0].var.same_as(second.iter_vars[0].var)
    assert first.body.value.same_as(first.iter_vars[0].var)
    assert second.body.value.same_as(second.iter_vars[0].var)
    # Shared input ownership must protect both original occurrences.
    assert block.iter_vars[0].var.same_as(var)
    assert block.body.value.same_as(var)


def test_shared_buffer_parameter_regions_across_functions():
    """Parameter renaming reaches both block regions and buffer accesses."""
    n = tirx.Var("n", "int32")
    buffer = tirx.decl_buffer((n,), "float32", "buffer")
    region = tirx.BufferRegion(buffer, [tvm.ir.Range(0, n)])
    block = s_tir.SBlock([], [region], [], "root", tirx.Evaluate(tirx.BufferLoad(buffer, [0])))
    func = tirx.PrimFunc([buffer], s_tir.SBlockRealize([], True, block))
    before = tvm.IRModule({"first": func, "second": func})

    after = s_tir.transform.ConvertSSA()(before)

    first, second = after["first"], after["second"]
    assert not first.params[0].same_as(second.params[0])
    for updated in [first, second]:
        updated_block = updated.body.block
        assert updated_block.reads[0].source.same_as(updated.params[0])
        assert updated_block.body.value.source.same_as(updated.params[0])
        assert updated_block.reads[0].region[0].extent.same_as(updated.params[0].ty.shape[0])


if __name__ == "__main__":
    tvm.testing.main()
