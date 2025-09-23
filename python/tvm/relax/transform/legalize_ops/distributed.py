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
# pylint: disable=invalid-name
"""Default legalization function for distir-related operators."""
from tvm import tir, relax
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from ...op import call_pure_packed
from ...struct_info import ShapeStructInfo
from .common import register_legalize


@register_legalize("relax.dist.redistribute_replica_to_shard")
def _redistribute_replica_to_shard(_bb: BlockBuilder, call: Call) -> Expr:
    num_workers = call.attrs.num_workers
    axis = call.attrs.axis
    worker_id_symbol = tir.Var("worker_id", "int64")
    worker_id_var = _bb.emit(
        call_pure_packed("runtime.disco.worker_id", sinfo_args=[ShapeStructInfo(None)])
    )
    _bb.match_cast(worker_id_var, ShapeStructInfo([worker_id_symbol]))

    split_axis_size = call.args[0].struct_info.shape[axis]
    return relax.op.strided_slice(
        call.args[0],
        axes=[axis],
        begin=[worker_id_symbol * split_axis_size // num_workers],
        end=[(worker_id_symbol + 1) * split_axis_size // num_workers],
        assume_inbound=True,
    )
