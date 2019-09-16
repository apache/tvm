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

        self.l = self.mod.get_global_type_var("List")
        list_adt = self.mod[self.l]
        self.cons = list_adt.constructors[0]
        self.nil = list_adt.constructors[1]

        self.optional = self.mod.get_global_type_var("Option")
        optional_adt = self.mod[self.optional]
        self.some = optional_adt.constructors[0]
        self.none = optional_adt.constructors[1]

        self.tree = self.mod.get_global_type_var("Tree")
        tree_adt = self.mod[self.tree]
        self.rose = tree_adt.constructors[0]

        GLOBAL_DEFS = [
            "id",
            "compose",
            "flip",
            "hd",
            "tl",
            "nth",
            "update",
            "map",
            "foldl",
            "foldr",
            "foldr1",
            "concat",
            "filter",
            "zip",
            "rev",
            "map_accuml",
            "map_accumr",
            "unfoldl",
            "unfoldr",
            "sum",
            "length",
            "tmap",
            "size",
            "iterate",
        ]
        for global_def in GLOBAL_DEFS:
            setattr(self, global_def, self.mod.get_global_var(global_def))
