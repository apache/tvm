
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

#include <tvm/parser/parser.h>
#include <tvm/relay/qnn/transform.h>

#include "te_compiler.h"

namespace tvm {
namespace relay {
namespace backend {

TVM_REGISTER_NODE_TYPE(StorageInfoNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StorageInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = ref.as<StorageInfoNode>();
      p->stream << "StorageInfoNode("
                << "storage_ids=[";
      for (auto id : node->storage_ids) {
        p->stream << id << ",";
      }
      p->stream << "], virtual_devices=[";
      for (const auto& virtual_device : node->virtual_devices) {
        p->stream << virtual_device << ",";
      }
      p->stream << "], storage_size_in_bytes=[";
      for (auto bytes : node->storage_sizes_in_bytes) {
        p->stream << bytes << ",";
      }
      p->stream << "])";
    });

StorageInfo::StorageInfo(std::vector<int64_t> storage_ids,
                         std::vector<VirtualDevice> virtual_devices,
                         std::vector<int64_t> storage_sizes_in_bytes) {
  ICHECK_EQ(storage_ids.size(), virtual_devices.size());
  ICHECK_EQ(storage_ids.size(), storage_sizes_in_bytes.size());
  auto node = make_object<StorageInfoNode>();
  node->storage_ids = std::move(storage_ids);
  node->virtual_devices = std::move(virtual_devices);
  node->storage_sizes_in_bytes = std::move(storage_sizes_in_bytes);
  data_ = std::move(node);
}

// This is the legacy interface for devices as DLDeviceTypes (represented by integers)
TVM_REGISTER_GLOBAL("relay.ir.StorageInfo")
    .set_body_typed([](const Array<Integer>& sids, const Array<Integer>& device_types,
                       const Array<Integer>& sizes_in_bytes) {
      std::vector<int64_t> sids_v;
      sids_v.reserve(sids.size());
      for (auto s : sids) {
        sids_v.push_back(s);
      }
      std::vector<VirtualDevice> virtual_devices_v;
      virtual_devices_v.reserve(device_types.size());
      for (const auto& device_type : device_types) {
        virtual_devices_v.emplace_back(VirtualDevice::ForDeviceType(device_type));
      }
      std::vector<int64_t> size_in_bytes_v;
      size_in_bytes_v.reserve(sizes_in_bytes.size());
      for (auto s : sizes_in_bytes) {
        size_in_bytes_v.push_back(s);
      }
      return StorageInfo(std::move(sids_v), std::move(virtual_devices_v),
                         std::move(size_in_bytes_v));
    });

TVM_REGISTER_GLOBAL("relay.ir.StorageInfoStorageIds").set_body_typed([](StorageInfo si) {
  Array<tvm::Integer> ids;
  for (auto id : si->storage_ids) {
    ids.push_back(id);
  }
  return ids;
});

// This is the legacy interface for devices as DLDeviceTypes (represented by integers)
TVM_REGISTER_GLOBAL("relay.ir.StorageInfoDeviceTypes").set_body_typed([](StorageInfo si) {
  Array<tvm::Integer> device_types;
  for (const auto& virtual_device : si->virtual_devices) {
    device_types.push_back(virtual_device->device_type());
  }
  return device_types;
});

TVM_REGISTER_GLOBAL("relay.ir.StorageInfoStorageSizes").set_body_typed([](StorageInfo si) {
  Array<tvm::Integer> storage_sizes_in_bytes;
  for (auto id : si->storage_sizes_in_bytes) {
    storage_sizes_in_bytes.push_back(id);
  }
  return storage_sizes_in_bytes;
});

TVM_REGISTER_NODE_TYPE(StaticMemoryPlanNode);

StaticMemoryPlan::StaticMemoryPlan(Map<Expr, StorageInfo> expr_to_storage_info) {
  auto n = make_object<StaticMemoryPlanNode>();
  n->expr_to_storage_info = std::move(expr_to_storage_info);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relay.ir.StaticMemoryPlan")
    .set_body_typed([](const Map<Expr, StorageInfo>& expr_to_storage_info) {
      return StaticMemoryPlan(expr_to_storage_info);
    });

// TODO(mbs): Cf GetMemorySizeBytes in aot_executor_codegen.cc, GetMemorySize in
// graph_plan_memory.cc
int64_t CalculateRelayExprSizeBytes(const Type& expr_type) {
  if (expr_type->IsInstance<TupleTypeNode>()) {
    auto tuple_type = Downcast<TupleType>(expr_type);
    int64_t size = 0;
    for (const auto& field : tuple_type->fields) {
      size += CalculateRelayExprSizeBytes(field);
    }
    return size;
  }
  if (expr_type->IsInstance<FuncTypeNode>()) {
    return 0;
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

FunctionInfo::FunctionInfo(Map<Target, Integer> workspace_sizes, Map<Target, Integer> io_sizes,
                           Map<Target, Integer> constant_sizes,
                           Map<Target, tir::PrimFunc> tir_primfuncs,
                           Map<Target, Function> relay_primfuncs) {
  ObjectPtr<FunctionInfoNode> n = make_object<FunctionInfoNode>();
  n->workspace_sizes = std::move(workspace_sizes);
  n->io_sizes = std::move(io_sizes);
  n->constant_sizes = std::move(constant_sizes);
  n->tir_primfuncs = std::move(tir_primfuncs);
  n->relay_primfuncs = std::move(relay_primfuncs);
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FunctionInfoNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FunctionInfoNode*>(ref.get());
      p->stream << "FunctionInfoNode(\n"
                << "workspace_sizes=" << node->workspace_sizes << ",\n  io_sizes=" << node->io_sizes
                << ",\n  constant_sizes=" << node->constant_sizes
                << ",\n  tir_primfuncs=" << node->tir_primfuncs
                << ",\n  relay_primfuncs=" << node->relay_primfuncs << ")";
    });

ExecutorCodegenMetadata::ExecutorCodegenMetadata(
    Array<tir::Var> inputs, Array<tir::Var> pools, Array<String> devices, Array<String> outputs,
    String executor, String mod_name, String interface_api, bool unpacked_api,
    Map<tir::Var, tir::usmp::AllocatedPoolInfo> pool_inputs) {
  auto n = make_object<ExecutorCodegenMetadataNode>();
  n->inputs = inputs;
  n->pools = pools;
  n->devices = devices;
  n->outputs = outputs;
  n->executor = executor;
  n->interface_api = interface_api;
  n->unpacked_api = unpacked_api;
  n->mod_name = mod_name;
  n->pool_inputs = pool_inputs;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ExecutorCodegenMetadataNode);

Array<Pass> GetPassPrefix(bool is_homegeneous, bool is_vm) {
  Array<Pass> pass_seqs;
  // TODO(mbs): Would be nice to get spans on all diagnostics, but since they arg forgotton
  // by most passes there's little utility in including this now. Plus we'd need to only do
  // this if there's no existing spans to work from.
  // pass_seqs.push_back(parser::AnnotateSpans());
  Array<runtime::String> entry_functions{"main"};
  pass_seqs.push_back(transform::RemoveUnusedFunctions(entry_functions));
  pass_seqs.push_back(transform::ToBasicBlockNormalForm());
  // Run all dialect legalization passes.
  pass_seqs.push_back(relay::qnn::transform::Legalize());

  // Legalize pass is restricted to homogeneous execution for now.
  if (is_homegeneous) {
    pass_seqs.push_back(transform::Legalize());
  }

  pass_seqs.push_back(transform::SimplifyInference());

  if (is_vm) {
    // eta expand to support constructors in argument position
    pass_seqs.push_back(transform::EtaExpand(
        /* expand_constructor */ true, /* expand_global_var */ false));
  } else {
    // Convert Dynamic ops to static versions
    pass_seqs.push_back(transform::DynamicToStatic());
  }

  PackedFunc fskip = PackedFunc([](TVMArgs args, TVMRetValue* rv) {
    Expr expr = args[0];
    if (expr.as<CallNode>()) {
      auto call_node = expr.as<CallNode>();
      auto op_node = call_node->op.as<OpNode>();
      if (op_node->name == "cast") {
        auto attrs = call_node->attrs.as<CastAttrs>();
        if (attrs->dtype == DataType::Int(32)) {
          *rv = true;
        }
      }
    }
    *rv = false;
  });
  pass_seqs.push_back(transform::EliminateCommonSubexpr(fskip));
  pass_seqs.push_back(transform::SimplifyExpr());
  pass_seqs.push_back(transform::CombineParallelConv2D(3));
  pass_seqs.push_back(transform::CombineParallelDense(3));
  pass_seqs.push_back(transform::CombineParallelBatchMatmul(3));
  pass_seqs.push_back(transform::FoldConstant());
  pass_seqs.push_back(transform::FoldScaleAxis());
  pass_seqs.push_back(transform::CanonicalizeCast());
  pass_seqs.push_back(transform::CanonicalizeOps());

  // Alter layout transformation is currently only applied to homogeneous execution.
  if (is_homegeneous) {
    if (!is_vm) {
      pass_seqs.push_back(transform::InferType());
    }
    pass_seqs.push_back(transform::AlterOpLayout());
  }

  // Fast math optimizations.
  pass_seqs.push_back(transform::FastMath());
  pass_seqs.push_back(transform::FoldConstant());
  return pass_seqs;
}

std::unordered_map<Target, IRModule, TargetStrHash, TargetStrEqual>
TargetModuleMapToTargetStrModuleMap(Map<Target, IRModule> input_map) {
  std::unordered_map<Target, IRModule, TargetStrHash, TargetStrEqual> std_map;
  for (auto kv : input_map) {
    std_map[kv.first] = kv.second;
  }
  return std_map;
}

Map<Target, IRModule> TargetStrModuleMapToTargetModuleMap(
    std::unordered_map<Target, IRModule, TargetStrHash, TargetStrEqual> input_map) {
  Map<Target, IRModule> tvm_map;
  for (auto kv : input_map) {
    tvm_map.Set(kv.first, kv.second);
  }
  return tvm_map;
}

void UpdateAutoSchedulerOpWeights(const IRModule& module) {
  const auto* te_compiler_update_weights =
      runtime::Registry::Get("auto_scheduler.relay_integration.te_compiler_update_weights");

  ICHECK(te_compiler_update_weights != nullptr)
      << "auto_scheduler.relay_integration.te_compiler_update_weights";

  Map<String, Integer> weight_map =
      module->GetAttr<Map<String, Integer>>("op_weights", Map<String, Integer>()).value();

  (*te_compiler_update_weights)(weight_map);
}

}  // namespace backend
}  // namespace relay
}  // namespace tvm
