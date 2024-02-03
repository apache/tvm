/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "utils.h"

#include <vector>
namespace tvm {
namespace relax {
namespace distributed {

Array<distributed::DTensorStructInfo> GetInputDTensorStructInfo(const Call& call,
                                                                const BlockBuilder& ctx) {
  Op op = Downcast<Op>(call->op);
  Array<Expr> args = GetCallArgs(call);
  Array<distributed::DTensorStructInfo> input_tensor_sinfo;
  input_tensor_sinfo.reserve(args.size());
  for (const Expr& arg : args) {
    const auto* sinfo = GetStructInfoAs<distributed::DTensorStructInfoNode>(arg);
    if (sinfo != nullptr) {
      input_tensor_sinfo.push_back(GetRef<distributed::DTensorStructInfo>(sinfo));
    }
  }
  return input_tensor_sinfo;
}

StructInfo InferShardingSpec(const Call& call, const BlockBuilder& ctx,
                             const StructInfo& orig_output_sinfo,
                             distributed::FBuildAxisGraph f_build_graph) {
  Array<distributed::DTensorStructInfo> input_dtensor_sinfos = GetInputDTensorStructInfo(call, ctx);
  for (int i = 1; i < static_cast<int>(input_dtensor_sinfos.size()); i++) {
    ICHECK(StructuralEqual()(input_dtensor_sinfos[0]->device_mesh,
                             input_dtensor_sinfos[i]->device_mesh));
  }
  distributed::DeviceMesh device_mesh = input_dtensor_sinfos[0]->device_mesh;
  Var output_var("output", orig_output_sinfo);
  distributed::AxisGroupGraph axis_group_graph;
  f_build_graph(output_var, call, &axis_group_graph);
  Array<Expr> args = GetCallArgs(call);
  int n_input_var = input_dtensor_sinfos.size();
  for (int i = 0; i < n_input_var; i++) {
    distributed::DTensorStructInfo dtensor_sinfo = input_dtensor_sinfos[i];
    Expr input_tensor = args[i];
    for (int j = 0; j < static_cast<int>(device_mesh->shape.size()); j++) {
      distributed::PlacementSpec placement_spec = dtensor_sinfo->placement->dim_specs[j];
      if (placement_spec->kind != distributed::PlacementSpecKind::kSharding) {
        continue;
      }
      axis_group_graph.AddSrcShardingPoint({input_tensor.get(), placement_spec->axis},
                                           {dtensor_sinfo->device_mesh, j});
    }
  }
  axis_group_graph.PropagateShardingSpec();
  Array<TensorStructInfo> orig_output_tensor_sinfos;
  if (const auto* tensor_sinfo = orig_output_sinfo.as<TensorStructInfoNode>()) {
    orig_output_tensor_sinfos.push_back(GetRef<TensorStructInfo>(tensor_sinfo));
  } else {
    const auto* tuple_sinfo = orig_output_sinfo.as<TupleStructInfoNode>();
    ICHECK(tuple_sinfo);
    for (const auto& sinfo : tuple_sinfo->fields) {
      orig_output_tensor_sinfos.push_back(Downcast<TensorStructInfo>(sinfo));
    }
  }
  Array<StructInfo> new_output_dtensor_sinfos;
  for (int idx = 0; idx < static_cast<int>(orig_output_tensor_sinfos.size()); idx++) {
    Array<distributed::PlacementSpec> output_placement_specs(
        std::vector<distributed::PlacementSpec>(device_mesh->shape.size(),
                                                distributed::PlacementSpec::Replica()));
    for (int i = 0; i < orig_output_tensor_sinfos[idx]->ndim; i++) {
      distributed::AxisShardingSpec sharding_spec;
      bool has_sharding_spec;
      std::tie(sharding_spec, has_sharding_spec) =
          axis_group_graph.GetAxisShardingSpec({output_var.get(), i, idx});
      if (has_sharding_spec) {
        output_placement_specs.Set(sharding_spec.second, distributed::PlacementSpec::Sharding(i));
      }
    }
    new_output_dtensor_sinfos.push_back(
        DTensorStructInfo(orig_output_tensor_sinfos[idx], device_mesh,
                          distributed::Placement(output_placement_specs)));
  }

  return new_output_dtensor_sinfos.size() == 1 ? new_output_dtensor_sinfos[0]
                                               : TupleStructInfo(new_output_dtensor_sinfos);
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
