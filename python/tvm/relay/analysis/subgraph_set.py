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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, unused-import
"""Subgraphs used in Relay."""

from tvm.runtime import Object
from . import _ffi_api


class SubgraphSet(Object):
    """Class to represent a relay expression split into subgraphs."""

    def __init__(self, expr, subgraph_begin_op, subgraph_end_op):
        """Construct subgraphs from an expression.

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression from which to construct the subgraphs.
        subgraph_begin_op : tvm.relay.Op
            The subgraph begin annotation.
        subgraph_end_op : tvm.relay.Op
            The subgraph end annotation.

        """
        self.__init_handle_by_constructor__(_ffi_api.SubgraphSet,
                                            expr,
                                            subgraph_begin_op,
                                            subgraph_end_op)

    def __len__(self):
        return len(self.subgraphs)

    def get_subgraph(self, expr):
        """Get the subgraph an expression belongs to.

        Parameters
        ----------
        expr : tvm.relay.Expr
            The expression.

        Returns
        -------
        subgraph : Subgraph
            The subgraph containing the expression.
            None if not found.
        """
        return _ffi_api.GetSubgraph(self, expr)
