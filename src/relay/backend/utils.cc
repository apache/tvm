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

/*!
 * \file relay/backend/util.cc
 * \brief Relay backend utilities.
 */

#include "utils.h"

namespace tvm {
namespace relay {
namespace backend {

TVM_REGISTER_NODE_TYPE(StorageInfoNode);

StorageInfo::StorageInfo(std::vector<int64_t> storage_ids, std::vector<DLDeviceType> device_types,
                         std::vector<int64_t> storage_sizes_in_bytes) {
  auto n = make_object<StorageInfoNode>();
  n->storage_ids = std::move(storage_ids);
  n->device_types = std::move(device_types);
  n->storage_sizes_in_bytes = std::move(storage_sizes_in_bytes);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(StaticMemoryPlanNode);

StaticMemoryPlan::StaticMemoryPlan(Map<Expr, StorageInfo> expr_to_storage_info) {
  auto n = make_object<StaticMemoryPlanNode>();
  n->expr_to_storage_info = std::move(expr_to_storage_info);
  data_ = std::move(n);
}

int64_t CalculateRelayExprSizeBytes(const Type& expr_type) {
  if (expr_type->IsInstance<TupleTypeNode>()) {
    auto tuple_type = Downcast<TupleType>(expr_type);
    int64_t size = 0;
    for (const auto& field : tuple_type->fields) {
      size += CalculateRelayExprSizeBytes(field);
    }
    return size;
  }
  auto tensor_type = expr_type.as<TensorTypeNode>();
  auto shape = tensor_type->shape;
  int num_of_elements = 1;
  for (const auto& dim_index_expr : shape) {
    if (dim_index_expr->IsInstance<IntImmNode>()) {
      num_of_elements *= dim_index_expr.as<IntImmNode>()->value;
    } else {
      // If shape is dynamic, we cannot calculate workspace in compile time.
      num_of_elements = 0;
    }
  }
  auto element_size = tensor_type->dtype.bytes();
  return element_size * num_of_elements;
}

TVM_REGISTER_NODE_TYPE(FunctionInfoNode);

}  // namespace backend
}  // namespace relay
}  // namespace tvm
