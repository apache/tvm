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
"""Insert annotation.compiler_begin/compiler_end according to the
coloring of the Relay IR nodes
"""
import tvm
from tvm.relay.op.annotation import compiler_begin, compiler_end


class AnnotateForRelayCompiler(tvm.relay.ExprMutator):
    """Annotate the graph with `annotation.compiler_begin` and `annotation.compiler_end`

    Parameters
    ----------
    options: dict
        The partitioner option dict

    edm: ExportDecisionMaker
        A object returning True/False about whether a Relay node should be exported

    """

    def __init__(self, options, edm):
        super().__init__()
        self._options = options
        self._compiler = self._options["tvm"]["external_compiler"]
        self._edm = edm
        self._in_graph = False

    def annotate(self, func):
        """Annotate the graph with `annotation.compiler_begin` and `annotation.compiler_end`

        Parameters
        ----------
        func: tvm.relay.Function
            The function to be annotated

        Returns
        -------
        func: tvm.relay.Function
            The annotated function

        """
        assert isinstance(func, tvm.relay.Function)
        return self.visit(func)

    def visit(self, expr):
        export_result = self._edm.node_is_exported(expr, self._compiler)
        if export_result == self._edm.EXPORT_RESULT["YES"]:
            if not self._in_graph:
                self._in_graph = True
                new_expr = super().visit(expr)
                assert self._in_graph
                self._in_graph = False  # subgraph should exit here when returning from children
                return compiler_end(new_expr, self._compiler)
        elif export_result == self._edm.EXPORT_RESULT["NO"]:
            if self._in_graph:
                self._in_graph = False
                new_expr = super().visit(expr)
                assert not self._in_graph
                self._in_graph = True  # restore `self._in_graph` in case other siblings needs it
                return compiler_begin(new_expr, self._compiler)

        ret = super().visit(expr)
        return ret
