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
"""FlattenBuffer must keep buffer identity coherent across rebuilds.

Rebuilding a buffer mints a fresh typed variable, so every remaining
reference — including loads embedded in another buffer's type fields and
loads spliced into indices by the elem_offset fold — must be remapped to
the rebuilt identity, or SplitHostDevice later sees them as undefined
and hoists dead variables into the kernel ABI.
"""

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.tirx.transform import FlattenBuffer


def _collect_defined_buffers(func):
    defined = set()

    def visit(node):
        if isinstance(node, tvm.tirx.AllocBuffer | tvm.tirx.DeclBuffer):
            defined.add(node.buffer)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return defined


def _assert_loads_reference_defined_buffers(func):
    """Every BufferLoad — direct, or embedded in a buffer's type fields —
    must reference a buffer defined by an AllocBuffer/DeclBuffer in the
    function."""
    defined = _collect_defined_buffers(func)

    def is_defined(buf):
        return any(buf.same_as(d) for d in defined)

    stale = []

    def check_expr(expr, where):
        def visit(node):
            if isinstance(node, tvm.ir.TensorLoad) and not is_defined(node.source):
                stale.append(f"{where}: load of {node.source.name}")

        tvm.tirx.stmt_functor.post_order_visit(expr, visit)

    def visit(node):
        if isinstance(node, tvm.ir.TensorLoad | tvm.tirx.BufferStore):
            buffer = node.source if isinstance(node, tvm.ir.TensorLoad) else node.buffer
            if not is_defined(buffer):
                stale.append(f"access of {buffer.name}")
            for index in node.indices:
                check_expr(index, f"index of {buffer.name}")
        if isinstance(node, tvm.tirx.AllocBuffer | tvm.tirx.DeclBuffer):
            for extent in node.buffer.shape:
                check_expr(extent, f"shape of {node.buffer.name}")
            if node.buffer.elem_offset is not None:
                check_expr(node.buffer.elem_offset, f"elem_offset of {node.buffer.name}")

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert not stale, f"stale buffer references after FlattenBuffer: {stale}"


def _flatten(func):
    mod = tvm.IRModule({"main": func})
    with tvm.target.Target("cuda"):
        return next(iter(FlattenBuffer()(mod).functions_items()))[1]


def test_flatten_remaps_loads_in_view_shape():
    """A view sized by a local scalar: the scalar's rebuild must reach the
    load embedded in the view's shape."""

    @T.prim_func(private=True)
    def before():
        n = T.alloc_local([1], "int32")
        n[0] = 8
        data = T.alloc_buffer([64], "float16", scope="shared")
        view = T.decl_buffer((n[0],), "float16", data.data, scope="shared")
        view[0] = T.float16(0)

    _assert_loads_reference_defined_buffers(_flatten(before))


def test_flatten_remaps_loads_in_folded_elem_offset():
    """A view offset by a local scalar: the elem_offset fold splices the
    scalar load into every access index; those spliced loads must follow
    the scalar's rebuild."""

    @T.prim_func(private=True)
    def before():
        n = T.alloc_local([1], "int32")
        n[0] = 4
        base = T.alloc_buffer([128], "uint64", scope="shared")
        mbar = T.decl_buffer((1,), "uint64", base.data, elem_offset=n[0], scope="shared")
        mbar[0] = T.uint64(1)

    after = _flatten(before)
    _assert_loads_reference_defined_buffers(after)

    # The fold must actually have spliced the offset into the store index.
    found = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.name.startswith("mbar"):

            def inner(sub):
                if isinstance(sub, tvm.ir.TensorLoad):
                    found.append(sub)

            tvm.tirx.stmt_functor.post_order_visit(node.indices[0], inner)

    tvm.tirx.stmt_functor.post_order_visit(after.body, visit)
    assert found, "expected the folded elem_offset load in the mbar store index"


def test_flatten_keeps_identity_of_already_flat_buffers():
    """A flat buffer whose type is unchanged by flattening must keep its
    identity (no gratuitous rebuild)."""

    @T.prim_func(private=True)
    def before():
        flat = T.alloc_buffer([32], "float32", scope="shared", layout=None)
        flat[0] = T.float32(0)

    before_allocs = {}

    def collect_before(node):
        if isinstance(node, tvm.tirx.AllocBuffer):
            before_allocs[node.buffer.name] = node.buffer

    tvm.tirx.stmt_functor.post_order_visit(before.body, collect_before)

    after = _flatten(before)
    preserved = []

    def visit(node):
        if isinstance(node, tvm.tirx.AllocBuffer) and node.buffer.name in before_allocs:
            preserved.append(node.buffer.same_as(before_allocs[node.buffer.name]))

    tvm.tirx.stmt_functor.post_order_visit(after.body, visit)
    assert preserved and all(preserved), "already-flat buffer identity was not preserved"


if __name__ == "__main__":
    tvm.testing.main()
