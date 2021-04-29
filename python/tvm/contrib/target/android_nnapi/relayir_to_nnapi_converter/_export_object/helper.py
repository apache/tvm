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
"""Namespace for helper objects/methods that's not part of the JSON
content. This includes the symbol table, checking methods, ...
"""
from .operand import Operand as _Operand


class Helper:
    def __init__(self, export_obj):
        self._export_obj = export_obj
        self.node_to_operand_idxs_map = {}
        self.type_to_idx_map = {}
        self.operand = _Operand(self._export_obj)
