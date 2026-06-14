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

"""Reduction dispatch variant registrations."""

from tvm.tirx.operator.tile_primitive import register_dispatch
from tvm.tirx.operator.tile_primitive.common import ReduceOpType

from .utils import reduction_trn

for _op_name, _op_type in {
    "sum": ReduceOpType.SUM,
    "max": ReduceOpType.MAX,
    "min": ReduceOpType.MIN,
}.items():

    @register_dispatch(_op_name, "trn", variant="reduction", priority=0)
    def _reduction_dispatch(op, sctx, _ty=_op_type):
        return reduction_trn(op, _ty, sctx)
