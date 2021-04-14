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
 * \file annotate_texture_storage.cc
 * \brief Collection of target specific relay passes which
 * storage scope related information.
 *
 *  - CollectStorageInfo returns a mapping from relay expr
 *    to a list of output storage scopes for each output.
 *    These scopes are used during memory planning as well
 *    as downstream when doing codegen (see CollectBufferBinds)
 *    and in the graph runtime when doing runtime dataspace
 *    allocations.
 *
 *  - CollectBufferBinds returns an array of tir::Buffer given
 *    the storage info yielded from CollectStogrageInfo. These
 *    buffers are bound to tensors created by the compile engine
 *    and are used as binds when calling tvm::lower/build.
 *
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/relay/attrs/nn.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {
namespace {

class StorageInfo : private ExprVisitor{
 public:
  StorageInfo(const Map<Expr, Integer>& dev_map, const Map<Integer, Target>& target_map)
    : device_ids_(dev_map), targets_(target_map) {;}
  static Map<Expr, Array<String>> GetStorageMap(const Expr& expr,
                                                const Map<Expr, Integer>& dev_map,
                                                const Map<Integer, Target>& target_map) {
    StorageInfo storage_info(dev_map, target_map);
    storage_info.Visit(expr);
    storage_info.LegalizeProducerStorage();
    // TODO(csullivan): The below can be removed if either of the following are true:
    //   * Function outputs are persistent (can_realloc = False)
    //   * Runtime support is added for passing tensor shape through CopyFromTo API
    //     so that image pitch can be determined allowing the correct read to be
    //     enqueued from a texture pool.
    // For now we force write to global for the outputs of the function over which
    // memory planning will be performed. This should incur only a trivial change
    // in performance.
    storage_info.ForceGlobalOutputStorage(expr);
    Map<Expr, Array<String>> storage_map;
    for (auto& kv : storage_info.storage_scope_) {
      std::vector<String> storage_scopes;
      std::copy(kv.second.begin(), kv.second.end(), std::back_inserter(storage_scopes));
      storage_map.Set(GetRef<Expr>(kv.first), Array<String>{storage_scopes});
    }
    return storage_map;
  }

 private:
  void Visit(const Expr& expr) {
    // Pre-order traversal to enable upward propagation
    // of consumer storage scopes to producers when desirable.
    if (const auto* fn = expr.as<FunctionNode>()) {
      this->VisitExpr(fn->body);
      for (const auto& param : fn->params) {
        this->VisitExpr(param);
      }
    } else {
      this->VisitExpr(expr);
    }
  }

  void VisitExpr_(const VarNode* vn) final {
    ApplyConsumerScopeToInputs(vn);
  }

  void VisitExpr_(const ConstantNode* cn) final {
    ApplyConsumerScopeToInputs(cn, "weight");
  }

  void VisitExpr_(const CallNode* call) final {
    // Check the contents of this primitive function
    if (DeviceSupportsTextureStorage(GetRef<Expr>(call))) {
      if (const auto* fn = call->op.as<FunctionNode>()) {
        if (fn->HasNonzeroAttr(attr::kPrimitive)) {
          primitive_supports_texture_ = false;
          Visit(call->op);
          if (primitive_supports_texture_) {
            if (call->checked_type().as<TensorTypeNode>()) {
              storage_scope_[call].push_back("texture");
            } else {
              const auto* tuple_type = call->type_as<TupleTypeNode>();
              ICHECK(tuple_type);
              // TODO(csullivan): Add support for mixed output storage scope.
              // In current adreno storage planner all outputs of a
              // primitive function are assumed to be of the same storage
              // type. This should be easy to extend in the future.
              for (size_t i = 0; i < tuple_type->fields.size(); i++) {
                storage_scope_[call].push_back("texture");
              }
            }
          }
          // Add consumer storage scope information for call arguments
          for (auto& arg : call->args) {
            if (storage_scope_.count(call)) {
              ICHECK(!HasMixedStorageOutputs(call)) << "Mixed output storage scopes are not currently supported";
              consumer_storage_scopes_[arg.operator->()].push_back(storage_scope_[call][0]);
            } else {
              consumer_storage_scopes_[arg.operator->()].push_back("global");
            }
          }
        }
      }
    }

    primitive_supports_texture_ = SupportsTextureStorage(call);

    for (auto& arg : call->args) {
      Visit(arg);
    }
  }

  void ApplyConsumerScopeToInputs(const ExprNode* expr, std::string scope_suffix = "") {
    auto consumer_scopes_it = consumer_storage_scopes_.find(expr);
    if (consumer_scopes_it != consumer_storage_scopes_.end()) {
      std::string consumer_scope = GetConsumerScope(consumer_scopes_it->second);
      ICHECK(!storage_scope_.count(expr))
        << "Already propagated consumer scopes to input: " << GetRef<Expr>(expr);

      bool expr_is_rgba_vectorizable = false;
      if (const auto* ttype = expr->checked_type().as<TensorTypeNode>()) {
        auto inner_dim = ttype->shape.back().as<IntImmNode>();
        if (inner_dim && inner_dim->value == 4) {
          expr_is_rgba_vectorizable = true;
        }
      }

      // Only propagate texture scope from consumers to input expr if
      // the input shape of the input expr is rgba vectorizable.
      if (consumer_scope == "texture") {
        if (expr_is_rgba_vectorizable) {
          std::string scope = consumer_scope;
          // Apply any provided storage scope suffix before assignment
          if (!scope_suffix.empty()) {
            scope += (":" + scope_suffix);
          }
          storage_scope_[expr].push_back(scope);
        }
      } else {
        storage_scope_[expr].push_back(consumer_scope);
      }
    }
  }

  void LegalizeProducerStorage() {
    for (auto& kv : consumer_storage_scopes_) {
      const ExprNode* producer = kv.first;
      std::string legal_scope = GetConsumerScope(kv.second);
      if (storage_scope_.count(producer)) {
        ICHECK(!HasMixedStorageOutputs(producer)) << "Mixed output storage scopes are not currently supported";
        if (storage_scope_[producer][0].find(legal_scope) == std::string::npos) {
          for (size_t i = 0; i < storage_scope_[producer].size(); i++) {
            // Only support uniform storage scope accross all outputs for now
            storage_scope_[producer][i] = legal_scope;
          }
        }
      }
    }
  }

  void ForceGlobalOutputStorage(const Expr& expr) {
    // Mark function outputs as global scope
    if (const auto* func = expr.as<FunctionNode>()) {
      if (auto* tuple = func->body.as<TupleNode>()) {
        for (auto& field : tuple->fields) {
          if (storage_scope_.count(field.operator->())) {
            for (size_t i = 0; i < storage_scope_[field.operator->()].size(); i++) {
              storage_scope_[field.operator->()][i] = "global";
            }
          }
        }
      } else {
        if (storage_scope_.count(func->body.operator->())) {
          for (size_t i = 0; i < storage_scope_[func->body.operator->()].size(); i++) {
            storage_scope_[func->body.operator->()][i] = "global";
          }
        }
      }
    }
  }

  bool DeviceSupportsTextureStorage(const Expr& expr) {
    Target target;
    Integer dev_id{-1};
    if (device_ids_.count(expr) && targets_.count(device_ids_[expr])) {
      dev_id = device_ids_[expr];
      target = targets_[dev_id];
    } else if (targets_.size() == 1) {
      const auto& kv = targets_.begin();
      dev_id = (*kv).first;
      target = (*kv).second;
    }
    ICHECK(dev_id->value != -1) << "Error inferring target device, device mapping and targets do not match";
    Optional<String> t_device = target->GetAttr<String>("device");
    // Currently only `target = opencl --device=adreno` supports texture storage
    if (target->kind->device_type == kDLOpenCL && t_device.defined()) {
      if (t_device.value() == "adreno") { return true; }
    }
    return false;
  }

  std::string GetConsumerScope(const std::vector<std::string>& consumer_scopes) const {
    if (!consumer_scopes.size()) { return "global"; }
    std::string ref_scope = consumer_scopes[0];
    for (auto& consumer_scope : consumer_scopes) {
      if (consumer_scope != ref_scope) {
        return "global";
      }
    }
    return ref_scope;
  }

  bool HasMixedStorageOutputs(const ExprNode* expr) {
    if (storage_scope_.count(expr)) {
      std::string ref_scope = storage_scope_[expr][0];
      for (std::string& scope : storage_scope_[expr]) {
        if (scope != ref_scope) {
          return true;
        }
      }
    }
    return false;
  }

  bool SupportsTextureStorage(const CallNode* call) const {
    bool supports_texture_storage = false;
    if (auto attrs = call->attrs.as<Conv2DAttrs>()) {
      if (attrs->data_layout == "NCHW4c" && attrs->kernel_layout == "OIHW4o") {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<GlobalPool2DAttrs>()) {
      if (attrs->layout == "NCHW4c") {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<MaxPool2DAttrs>()) {
      if (attrs->layout == "NCHW4c") {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<AvgPool2DAttrs>()) {
      if (attrs->layout == "NCHW4c") {
        supports_texture_storage = true;
      }
    } else if (call->attrs.as<ConcatenateAttrs>()) {
      supports_texture_storage = true;
    } else if (auto attrs = call->attrs.as<LayoutTransformAttrs>()) {
      // Enable if either the source or destination layout is packed with vector length == 4.
      // Disabled for layout contraction due to a bug when writing from texture to global buffer.
      // TODO(csullivan): Enable proper code generation when emitting non-coalesced writes
      // of elements from a coalesced texture read.
      if ((attrs->dst_layout.find("4") == 4) /* || (attrs->src_layout.find("4") == 4) */) {
        supports_texture_storage = true;
      }
    }

    return supports_texture_storage;
  }

  /*! \brief expr device mapping */
  Map<Expr, Integer> device_ids_;
  /*! \brief device id to target mapping  */
  Map<Integer, Target> targets_;
  /*! \brief Temporary state for marking whether a visited function
   *         primitive supports texture storage scope */
  bool primitive_supports_texture_ = false;
  /*! \brief expr storage scope mapping for each output  */
  std::unordered_map<const ExprNode*, std::vector<std::string>> storage_scope_;
  /*! \brief output storage scopes used by consumers of expr key  */
  std::unordered_map<const ExprNode*, std::vector<std::string>> consumer_storage_scopes_;
};

String GetStorageScope(const Expr& expr, const Map<Expr, runtime::ADT>& storage_map, size_t output_index) {
  if (!storage_map.count(expr)) { return String{}; }
  auto storage_info = Downcast<Array<String>>(storage_map[expr][2]);
  if (output_index >= storage_info.size()) {
    return String{};
  }
  std::string scope = storage_info[output_index];
  auto pos = scope.find(":");
  if (pos != std::string::npos) {
    scope = scope.substr(0, pos);
  }
  return String(scope);
}
}

Array<tir::Buffer> CollectBufferBinds(const Call& call, const Map<Expr, runtime::ADT>& storage_map) {
  const auto* primfn = call->op.as<FunctionNode>();
  ICHECK(primfn);
  ICHECK(primfn->HasNonzeroAttr(attr::kPrimitive)) << "Can only collect buffer binds for primitive functions";
  ICHECK_EQ(call->args.size(), primfn->params.size()) << "Call arguments and function parameters do not match";

  auto make_buffer = [&storage_map](const Expr& expr, const TensorTypeNode* ttype, const std::string& name, size_t index = 0) {
    //String scope = GetStorageScope(expr, storage_map, index);
    auto storage_info = Downcast<Array<String>>(storage_map[expr][2]);
    std::string scope = "";
    if (storage_info.size()) {
      scope = storage_info[index];
    }

    PrimType storage_type(ttype->dtype);
    tir::Var var = GetStorageScope(expr, storage_map, index) == "texture" ? tir::Var(name, TextureType(storage_type)) : tir::Var(name, PointerType(storage_type));
    return tir::Buffer(var, ttype->dtype, ttype->shape, Array<PrimExpr>{}, Integer(0), name, scope, -1, 0, tir::BufferType::kDefault);
  };

  // Make input buffers
  Array<tir::Buffer> buffers;
  for (size_t i = 0; i < call->args.size(); i++) {
    const Expr& arg = call->args[i];
    if (const auto* ttype = primfn->params[i]->checked_type().as<TensorTypeNode>()) {
      buffers.push_back(make_buffer(arg, ttype, "placeholder" + std::to_string(i)));
    } else {
      const auto* tuple_type = primfn->params[i]->type_as<TupleTypeNode>();
      ICHECK(tuple_type);
      for (size_t j = 0; j < tuple_type->fields.size(); j++) {
        const auto* ttype = tuple_type->fields[j].as<TensorTypeNode>();
        ICHECK(ttype);
        buffers.push_back(make_buffer(arg, ttype, "placeholder" + std::to_string(i) + "_" + std::to_string(j), j));
      }
    }
  }

  // Make output buffers
  if (const auto* ttype = call->checked_type().as<TensorTypeNode>()) {
    buffers.push_back(make_buffer(call, ttype, "compute"));
  } else {
    const auto* tuple_type = call->type_as<TupleTypeNode>();
    ICHECK(tuple_type);
    for (size_t i = 0; i < tuple_type->fields.size(); i++) {
      const auto* ttype = tuple_type->fields[i].as<TensorTypeNode>();
      ICHECK(ttype);
      buffers.push_back(make_buffer(call, ttype, "compute" + std::to_string(i), i));
    }
  }

  return buffers;
}

Map<Expr, Array<String>> CollectTextureStorage(const Expr& expr,
                                               const Map<Expr, Integer>& dev_map,
                                               const Map<Integer, Target>& target_map) {
  return StorageInfo::GetStorageMap(expr, dev_map, target_map);
}

TVM_REGISTER_GLOBAL("relay.backend.opencl.adreno._CollectStorageInfo").set_body_typed(CollectTextureStorage);

TVM_REGISTER_GLOBAL("relay.backend.opencl.adreno._CollectBufferBinds").set_body_typed(CollectBufferBinds);

}  // namespace relay
}  // namespace tvm
