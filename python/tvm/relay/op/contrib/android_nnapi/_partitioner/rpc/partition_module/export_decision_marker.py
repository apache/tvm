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
"""Color Relay IR nodes to indicate the designated device of
execution."""
import tvm
from .platform_simulator import compute_device


class ExportDecisionMarker(tvm.relay.ExprVisitor):
    """A blackbox object telling whether a Relay node should be exported to the queried compiler.

    Parameters
    ----------
    options: dict
        The partitioner option dict.

    node_transfers: (Internal Format)
        The artifact of the partitioning algorithm.
    """

    EXPORT_RESULT = {
        "NO": 0,
        "YES": 1,
        "UNSURE": 2,
    }

    _DEVICE_COMPILERS = {
        compute_device.NnapiDevice.DEV_NAME: "android_nnapi",
        compute_device.TvmDevice.DEV_NAME: "tvm",
    }

    def __init__(self, options, node_transfers):
        super().__init__()
        self._options = options
        self._node_transfers = node_transfers
        assert (
            self._options["tvm"]["external_compiler"]
            == self._DEVICE_COMPILERS[compute_device.NnapiDevice.DEV_NAME]
        )

    def mark(self, func):
        assert isinstance(func, tvm.relay.Function)
        self._node_compiler_map = {func: compute_device.TvmDevice.DEV_NAME}
        self.memo_map[func] = None
        self._saved_devs = []
        self._parent_dev = compute_device.TvmDevice.DEV_NAME
        self.visit(func.body)

    def _set_parent(self, dev):
        self._saved_devs.append(self._parent_dev)
        self._parent_dev = dev

    def _restore_parent(self):
        self._parent_dev = self._saved_devs.pop()

    def node_is_exported(self, node, compiler):
        """Report whether a node is marked as exported.

        Parameters
        ----------
        node: tvm.relay.Node
            The queried node.

        compiler: str
            The compiler used to export.

        Returns
        -------
        exported: self.EXPORT_RESULT
            Whether the node is marked as exported with the compiler.
        """
        if isinstance(node, tvm.ir.Op):
            return self.EXPORT_RESULT["UNSURE"]

        verdict = self._node_compiler_map[node]
        if len(verdict) == 1 and verdict[0] == compiler:
            return self.EXPORT_RESULT["YES"]
        if compiler in verdict:
            return self.EXPORT_RESULT["UNSURE"]
        return self.EXPORT_RESULT["NO"]

    def visit(self, expr):
        if isinstance(expr, tvm.ir.Op):
            return super().visit(expr)

        next_dev = self._node_transfers[self._parent_dev][expr]
        next_compiler = self._DEVICE_COMPILERS[next_dev]
        self._node_compiler_map[expr] = [next_compiler]

        self._set_parent(next_dev)
        ret = super().visit(expr)
        self._restore_parent()
        return ret

    def visit_var(self, var):
        assert self._node_compiler_map[var] == ["tvm"]
        super().visit_var(var)

    def visit_let(self, let):
        raise NotImplementedError(let.type_key)

    def visit_function(self, fn):
        assert self._node_compiler_map[fn] == ["tvm"]
        super().visit_function(f)

    def visit_if(self, i):
        assert self._node_compiler_map[i] == ["tvm"]
        super().visit_if(i)

    def visit_global_var(self, gv):
        assert self._node_compiler_map[gv] == ["tvm"]
        super().visit_global_var(gv)

    def visit_ref_create(self, r):
        raise NotImplementedError(r.type_key)

    def visit_ref_read(self, r):
        raise NotImplementedError(r.type_key)

    def visit_ref_write(self, r):
        raise NotImplementedError(r.type_key)

    def visit_tuple_getitem(self, t):
        if isinstance(t.tuple_value, tvm.relay.Call):
            assert self._node_compiler_map[t] == ["tvm"]
        super().visit_tuple_getitem(t)

    def visit_constructor(self, c):
        raise NotImplementedError(c.type_key)

    def visit_match(self, m):
        raise NotImplementedError(m.type_key)
