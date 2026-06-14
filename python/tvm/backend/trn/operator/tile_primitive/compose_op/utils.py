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

"""Shared helpers for compose operator dispatches."""

from tvm.ir import Op
from tvm.tirx.operator.tile_primitive.common import ReduceOpType

# Operation code mappings
opcode_table = {
    Op.get("tirx.tile.add"): "add",
    Op.get("tirx.tile.sub"): "sub",
    Op.get("tirx.tile.mul"): "mul",
    Op.get("tirx.tile.maximum"): "max",
    Op.get("tirx.tile.minimum"): "min",
    Op.get("tirx.tile.sqrt"): "sqrt",
    Op.get("tirx.tile.sum"): "add",
    Op.get("tirx.tile.max"): "max",
    Op.get("tirx.tile.min"): "min",
    Op.get("tirx.tile.exp"): "exp",
}

optype_table = {
    Op.get("tirx.tile.sum"): ReduceOpType.SUM,
    Op.get("tirx.tile.max"): ReduceOpType.MAX,
    Op.get("tirx.tile.min"): ReduceOpType.MIN,
}
