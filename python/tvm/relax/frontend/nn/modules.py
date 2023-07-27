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
"""Builtin Modules."""
from typing import List, Optional

from tvm import relax as rx

from .core import Effect, Tensor


class IOEffect(Effect):
    """
    Modeling IO side effect, for example, printing the content of NDArrays on screen, inserting
    debug breakpoints, etc.
    """

    effect: Optional[rx.Var]

    def __init__(self):
        self.effect = None

    def emit_init(self, name_hint, builder: rx.BlockBuilder) -> List[rx.DataflowVar]:
        return [builder.emit(rx.op.null_value(), f"{name_hint}.io")]

    def create(self, name_hint: str) -> List[rx.Var]:
        assert self.effect is None
        self.effect = rx.Var(f"{name_hint}.io", struct_info=rx.ObjectStructInfo())
        return [self.effect]

    def finalize(self) -> List[rx.Var]:
        result = self.effect
        self.effect = None
        return [result]

    def print_(self, tensor: Tensor) -> None:
        """Encloses the side effect of NDArray printing"""
        raise NotImplementedError
