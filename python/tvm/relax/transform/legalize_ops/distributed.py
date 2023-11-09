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
from tvm import tir, te
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

    def te_R_to_S(tensor, worker_id):
        output_shape = list(tensor.shape)
        output_shape[axis] = output_shape[axis] // num_workers

        def index_func(out_indices):
            in_indices = []
            for i, idx in enumerate(out_indices):
                if i == axis:
                    in_indices.append(idx + worker_id * output_shape[axis])
                else:
                    in_indices.append(idx)
            return tuple(in_indices)

        return te.compute(
            output_shape, lambda *idx: tensor[index_func(idx)], name="redistribute_replica_to_shard"
        )

    return _bb.call_te(te_R_to_S, call.args[0], worker_id_symbol)
