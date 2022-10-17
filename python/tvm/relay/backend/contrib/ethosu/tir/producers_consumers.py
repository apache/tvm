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
# pylint: disable=invalid-name, unused-argument
"""The ProducersConsumers class"""
from typing import Optional
from collections.abc import KeysView
import tvm


class ProducersConsumers:
    """It associates pointers with the loop nest that produces
    their values and with the loop nest that consumes their values."""

    def __init__(self) -> None:
        self.indices: dict[tvm.tir.AttrStmt, int] = {}
        self.producers: list[(tvm.tir.AttrStmt, tvm.tir.expr.Var)] = []
        self.consumers: list[(tvm.tir.AttrStmt, list[tvm.tir.expr.Var])] = []
        self.allocate_variables: Optional[KeysView] = None

    def add_producer(self, var: tvm.tir.expr.Var, attr: tvm.tir.AttrStmt) -> None:
        """Add the attribute statement attr as producer of the variable var."""
        self.indices[attr] = len(self.producers)
        self.producers.append((attr, var))

    def get_producer(
        self, var: tvm.tir.expr.Var, attr: tvm.tir.AttrStmt
    ) -> Optional[tvm.tir.AttrStmt]:
        """Get the last attribute statement which produces the variable var when
        the current attribute statement is attr."""
        if var not in self.allocate_variables:
            return None

        index = self.indices[attr]
        for i in list(reversed(range(index + 1))):
            if self.producers[i][1] == var:
                return self.producers[i][0]
        return None

    def get_last_producer(self, var: tvm.tir.expr.Var) -> Optional[tvm.tir.AttrStmt]:
        """Get the last attribute statement which produces the variable var."""
        return self.get_producer(var, self.producers[-1][0])

    def add_allocate_variables(self, allocate_variables: KeysView) -> None:
        """Add the allocated variables."""
        self.allocate_variables = allocate_variables

    def add_consumer(self, var: tvm.tir.expr.Var, attr: tvm.tir.AttrStmt) -> None:
        """Add the attribute statement attr as consumer of the variable var."""
        index = self.indices[attr]
        if index < len(self.consumers):
            self.consumers[index][1].append(var)
        else:
            self.consumers.append((attr, [var]))

    def get_consumer(
        self, var: tvm.tir.expr.Var, attr: tvm.tir.AttrStmt
    ) -> Optional[tvm.tir.AttrStmt]:
        """Get the first attribute statement which consumes the variable var when
        the current attribute statement is attr."""
        index = self.indices[attr]
        for i in range(index, len(self.consumers)):
            if var in self.consumers[i][1]:
                return self.consumers[i][0]
        return None
