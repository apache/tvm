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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""A prelude containing useful global functions and ADT definitions."""
from .ty import GlobalTypeVar, TypeVar, FuncType, TupleType, scalar_type
from .expr import Var, Function, GlobalVar, Let, If, Tuple, TupleGetItem, const
from .op.tensor import add, subtract, equal
from .adt import Constructor, TypeData, Clause, Match
from .adt import PatternConstructor, PatternVar, PatternWildcard, PatternTuple
from .module import Module

class Prelude:
    """Contains standard definitions."""

    def __init__(self, mod=None):
        if mod is None:
            mod = Module()
        self.mod = mod
        self.load_prelude()

    def load_prelude(self):
        """Parses the Prelude from Relay's text format into a module."""
        # TODO(@jroesch): we should remove this helper when we port over prelude
        self.mod.import_from_std("prelude.rly")
        self.id = self.mod.get_global_var("id")
        self.compose = self.mod.get_global_var("compose")
