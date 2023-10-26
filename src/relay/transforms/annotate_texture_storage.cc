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
 *    to a map of storage scopes for each call argument.
 *    These scopes are used during memory planning as well
 *    as downstream when doing codegen and in the graph runtime when doing runtime dataspace
 *    allocations.
 *
 *  - AnnotateMemoryScope calls *target.CollectStorageInfo for all target been represented
 *    in the graph and rewrites graph modifying or inserting of VirtualDevice with required
 *    memory_scope collected from the CollectStorageInfo
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/expr.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "../op/memory/device_copy.h"
#include "../op/memory/memory.h"
#include "../transforms/device_aware_visitors.h"

namespace tvm {
namespace relay {
namespace {

/**
 * @brief Analyzes the graph and returns mapping of expressions vs desired memory scope
 */
class StorageInfo : private transform::DeviceAwareExprVisitor {
 public:
  StorageInfo() : transform::DeviceAwareExprVisitor(Optional<IRModule>()) {}

  static Map<Expr, Map<Expr, Array<String>>> GetStorageMap(const Expr& expr) {
    StorageInfo storage_info;
    storage_info.VisitExpr(expr);
    storage_info.LegalizeProducerStorage();
    Map<Expr, Map<Expr, Array<String>>> storage_map = storage_info.accept_textures_;
    for (auto& kv : storage_info.storage_scope_) {
      std::vector<String> storage_scopes;
      std::copy(kv.second.begin(), kv.second.end(), std::back_inserter(storage_scopes));
      Map<Expr, Array<String>> ent;
      ent.Set(Expr(), Array<String>{storage_scopes});
      storage_map.Set(GetRef<Expr>(kv.first), ent);
    }

    // Filling the input arguments by "global" scope to handle PlanDevice algo which propagates
    // virtual devices from outputs to inputs. At the same time outputs must be unconstrained
    // to avoid useless device_copy
    for (const auto& cs : storage_info.consumer_storage_scopes_) {
      // we have record in consumers that mean that potentially consumer
      // dealt with textures anyhow, it's safe to mark this expr as global scope
      // even without verification of the consumer's outputs scope
      if (storage_info.CanConsumeTextures(cs.second) &&
          storage_map.find(GetRef<Expr>(cs.first)) == storage_map.end()) {
        Map<Expr, Array<String>> ent;
        ent.Set(Expr(), Array<String>{"global"});
        storage_map.Set(GetRef<Expr>(cs.first), ent);
      }
    }

    // initial algo assumes mapping of outputs of the expr that is not enough, need to update
    // VirtualDevice for function variables to get proper codegen. Adding vars to storage_map
    for (const auto& a : storage_info.args_to_vars_) {
      if (storage_map.count(a.first)) {
        for (const auto& v : a.second) {
          if (storage_info.buffers_params.find(v) != storage_info.buffers_params.end()) {
            Map<Expr, Array<String>> ent;
            ent.Set(Expr(), Array<String>{"global"});
            storage_map.Set(v, ent);
          } else {
            storage_map.Set(v, storage_map[a.first]);
            if (storage_map[a.first][Expr()][0] == "global" &&
                storage_info.accept_textures_.count(v)) {
              Map<Expr, Array<String>> ent;
              ent.Set(Expr(), storage_info.accept_textures_[v][Expr()]);
              storage_map.Set(v, ent);
              for (const auto& calls : storage_info.accept_textures_[v]) {
                if (calls.first != Expr()) {
                  if (storage_map.count(a.first)) {
                    Map<Expr, Array<String>> ent_call = storage_map[a.first];
                    ent_call.Set(calls.first, calls.second);
                    storage_map.Set(a.first, ent_call);
                  } else {
                    Map<Expr, Array<String>> ent_call;
                    ent_call.Set(calls.first, calls.second);
                    storage_map.Set(a.first, ent_call);
                  }
                }
              }
            }
          }
        }
      }
    }
    return storage_map;
  }

 private:
  using transform::DeviceAwareExprVisitor::VisitExpr_;

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

  void VisitExpr_(const VarNode* vn) final { ApplyConsumerScopeToInputs(vn); }

  void VisitExpr_(const ConstantNode* cn) final { ApplyConsumerScopeToInputs(cn); }

  void DeviceAwareVisitExpr_(const CallNode* call) final {
    // Check the contents of this primitive function
    if (const auto* fn = call->op.as<FunctionNode>()) {
      if (fn->HasNonzeroAttr(attr::kPrimitive)) {
        primitive_supports_texture_ = false;
        Visit(call->op);
        if (primitive_supports_texture_) {
          if (call->checked_type().as<TensorTypeNode>()) {
            std::string scope = "global.texture";
            if (const auto* ttype = call->checked_type().as<TensorTypeNode>()) {
              scope = Scope(ttype->shape, GetVirtualDevice(GetRef<Expr>(call)));
            }
            storage_scope_[call].push_back(scope);
          } else {
            const auto* tuple_type = call->type_as<TupleTypeNode>();
            ICHECK(tuple_type);
            // TODO(csullivan): Add support for mixed output storage scope.
            // In current adreno storage planner all outputs of a
            // primitive function are assumed to be of the same storage
            // type. This should be easy to extend in the future.
            for (size_t i = 0; i < tuple_type->fields.size(); i++) {
              storage_scope_[call].push_back("global.texture");
            }
          }
          const int weights_pos = 1;
          for (size_t i = 0; i < fn->params.size(); i++) {
            args_to_vars_[call->args[i]].push_back(fn->params[i]);
            // adding info about arguments if they can be converted to texture
            for (const auto& ttype : FlattenTupleType(fn->params[i]->checked_type())) {
              std::string scope = Scope(ttype->shape, GetVirtualDevice(GetRef<Expr>(call)));
              if (expr_attrib.as<Conv2DAttrs>() || expr_attrib.as<Conv2DWinogradAttrs>()) {
                String kernel_layout = expr_attrib.as<Conv2DAttrs>()
                                           ? expr_attrib.as<Conv2DAttrs>()->kernel_layout
                                           : expr_attrib.as<Conv2DWinogradAttrs>()->kernel_layout;
                if ((i == weights_pos) && !ttype->dtype.is_float16() &&
                    CanUseBuffers(call->args[i], ttype->shape, kernel_layout)) {
                  buffers_params.insert(fn->params[i]);
                  buffers_args.insert(call->args[i]);
                  scope = "global";
                }
              }
              if (scope.find("global.texture") != std::string::npos) {
                if (accept_textures_.count(fn->params[i])) {
                  Map<Expr, Array<String>> ent = accept_textures_[fn->params[i]];
                  ent.Set(GetRef<Expr>(call), Array<String>{scope});
                  ent.Set(Expr(), Array<String>{scope});
                  accept_textures_.Set(fn->params[i], ent);
                } else {
                  Map<Expr, Array<String>> ent;
                  ent.Set(GetRef<Expr>(call), Array<String>{scope});
                  ent.Set(Expr(), Array<String>{scope});
                  accept_textures_.Set(fn->params[i], ent);
                }
              }
            }
          }
        }
        // Add consumer storage scope information for call arguments
        for (auto& arg : call->args) {
          if (storage_scope_.count(call)) {
            ICHECK(!HasMixedStorageOutputs(call))
                << "Mixed output storage scopes are not currently supported";
            consumer_storage_scopes_[arg.operator->()].push_back("global.texture");
          } else {
            consumer_storage_scopes_[arg.operator->()].push_back("global");
          }
        }
      }
    }
    if (!primitive_supports_texture_) {
      expr_attrib = call->attrs;
      primitive_supports_texture_ = SupportsTextureStorage(call);
    }

    for (auto& arg : call->args) {
      if (buffers_args.find(arg) == buffers_args.end()) {
        Visit(arg);
      }
    }
    // We have all callees filled into storage_scope_ if they support textures
    // We need to verify if this call expects texture and if it does not, remove from
    // storage_scope_ since initially storage_scope_ is filled only based on knowledge
    // that function able to work with textures, but not necessary that this texture is
    // expected by function callee
    for (auto& arg : call->args) {
      if (consumer_storage_scopes_.count(arg.operator->()) &&
          GetConsumerScope(consumer_storage_scopes_[arg.operator->()]) != "global.texture") {
        storage_scope_.erase(arg.operator->());
      }
    }
  }

  /**
   * Defines the name of the memory scope which can fit the tensor of required shape
   *
   * The scope stands for "global" if tensor does not satisfy current flattening rules for textures
   * (texture currently has to be 5d tensors with value eq 4 in the last dimension)
   *
   * The packing layout inside the texture scope (the part after the dash) is defined
   * during the shape itself. Hardware can have limitations on the texture spatial dimensions
   * we must not exceed these sizes. In addition to the fitting of h/w limitation we want to
   * get balanced packing where final spatial sizes of textures will not be too different
   * @param shape shape to be analyzed
   * @param vd VirtualDevice for the tensors determined of memory scope
   * @return string representing memory scope either "global" or "global.texture-layout"
   */
  std::string Scope(Array<PrimExpr> shape, const VirtualDevice& vd) {
    // currently we support only textures been made from 5d tensors
    // 5d requirement is not limitation of textures in general, it is limitation how
    // we are representing memory scopes/layout and flattening of textures in tir
    if (vd != VirtualDevice::FullyUnconstrained() && shape.size() == 5 &&
        shape[4].as<IntImmNode>()->value == 4) {
      std::map<int, std::string> diffs;
      int limit =
          vd->target->GetAttr<Integer>("texture_spatial_limit").value_or(Integer(16384))->value;
      int a0 = shape[0].as<IntImmNode>()->value;
      int a1 = shape[1].as<IntImmNode>()->value;
      int a2 = shape[2].as<IntImmNode>()->value;
      int a3 = shape[3].as<IntImmNode>()->value;

      int d3l = a0 * a1 * a2;
      int d3r = a3;
      int diff3 = d3l > d3r ? d3l - d3r : d3r - d3l;
      if (d3l < limit && d3r < limit) diffs[diff3] = "";

      int d2l = a0 * a1;
      int d2r = a2 * a3;
      int diff2 = d2l > d2r ? d2l - d2r : d2r - d2l;
      if (d2l < limit && d2r < limit) diffs[diff2] = "nhwc";

      int d1l = a0;
      int d1r = a1 * a2 * a3;
      int diff1 = d1l > d1r ? d1l - d1r : d1r - d1l;
      if (d1l < limit && d1r < limit) diffs[diff1] = "weight";
      if (!diffs.empty()) {
        std::string scope = "global.texture";
        if (!diffs.begin()->second.empty()) {
          scope += ("-" + diffs.begin()->second);
        }
        return scope;
      }
    }
    return "global";
  }

  void ApplyConsumerScopeToInputs(const ExprNode* expr) {
    std::string scope;
    auto consumer_scopes_it = consumer_storage_scopes_.find(expr);
    if (consumer_scopes_it != consumer_storage_scopes_.end()) {
      std::string consumer_scope = GetConsumerScope(consumer_scopes_it->second);
      ICHECK(!storage_scope_.count(expr))
          << "Already propagated consumer scopes to input: " << GetRef<Expr>(expr);

      bool expr_is_rgba_vectorizable = false;
      if (const auto* ttype = expr->checked_type().as<TensorTypeNode>()) {
        scope = Scope(ttype->shape, GetVirtualDevice(GetRef<Expr>(expr)));
        if (scope != "global") {
          auto inner_dim = ttype->shape.back().as<IntImmNode>();
          if (inner_dim && inner_dim->value == 4) {
            expr_is_rgba_vectorizable = true;
          }
        }
      }

      // Only propagate texture scope from consumers to input expr if
      // the input shape of the input expr is rgba vectorizable.
      if (consumer_scope.find("global.texture") != std::string::npos) {
        if (expr_is_rgba_vectorizable) {
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
        ICHECK(!HasMixedStorageOutputs(producer))
            << "Mixed output storage scopes are not currently supported";
        if (storage_scope_[producer][0].find(legal_scope) == std::string::npos) {
          for (size_t i = 0; i < storage_scope_[producer].size(); i++) {
            // Only support uniform storage scope across all outputs for now
            storage_scope_[producer][i] = legal_scope;
          }
        }
      }
    }
  }

  std::string GetConsumerScope(const std::vector<std::string>& consumer_scopes) const {
    if (!consumer_scopes.size()) {
      return "global";
    }
    std::string texture_tag = "global.texture";
    for (auto& consumer_scope : consumer_scopes) {
      if (consumer_scope.find(texture_tag) == std::string::npos) {
        return "global";
      }
    }
    return texture_tag;
  }

  bool CanConsumeTextures(const std::vector<std::string>& consumer_scopes) const {
    std::string texture_tag = "global.texture";
    for (auto& consumer_scope : consumer_scopes) {
      if (consumer_scope.find(texture_tag) == 0) {
        return true;
      }
    }
    return false;
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
    // we need to verify only entry functions since one of entry op defines main schedule
    for (const auto& arg : call->args) {
      if (!arg.as<VarNode>()) {
        return false;
      }
    }
    if (auto attrs = call->attrs.as<Conv2DAttrs>()) {
      if (attrs->data_layout == "NCHW4c" && attrs->kernel_layout == "OIHW4o") {
        supports_texture_storage = true;
      } else if (attrs->data_layout == "NHWC4c" &&
                 (attrs->kernel_layout == "HWOI4o" || attrs->kernel_layout == "HWIO4o" ||
                  attrs->kernel_layout == "OIHW4o")) {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<Conv2DWinogradAttrs>()) {
      if ((attrs->data_layout == "NCHW4c" || attrs->data_layout == "NHWC4c") &&
          (attrs->kernel_layout == "OIHW4o" || attrs->kernel_layout == "HWIO4o")) {
        supports_texture_storage = true;
      }
    } else if (auto attrs = call->attrs.as<Conv2DTransposeAttrs>()) {
      if (attrs->data_layout == "NCHW4c" && attrs->kernel_layout == "IOHW4o") {
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
    } else if (const OpNode* opnode = call->op.as<OpNode>()) {
      auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
      auto pattern = fpattern[GetRef<Op>(opnode)];
      if (pattern <= kCommReduce) {
        if (const auto* ttype = call->checked_type().as<TensorTypeNode>()) {
          if (ttype->shape.size() == 5) {
            auto node0 = ttype->shape[0].as<IntImmNode>();
            auto node1 = ttype->shape[1].as<IntImmNode>();
            auto node2 = ttype->shape[2].as<IntImmNode>();
            auto node3 = ttype->shape[3].as<IntImmNode>();
            auto node4 = ttype->shape[4].as<IntImmNode>();
            // if tensor has any dimension then textures are not supported
            if (!node0 || !node1 || !node2 || !node3 || !node4) {
              return false;
            }
            supports_texture_storage = true;
          }
        }
      }
    }

    return supports_texture_storage;
  }

  bool CanUseBuffers(const Expr param, const Array<PrimExpr> shape,
                     const String kernel_layout) const {
    bool use_buffer = false;
    if (param.as<ConstantNode>() && shape.size() == 5) {
      if (kernel_layout == "HWOI4o" || kernel_layout == "HWIO4o") {
        int a0 = shape[0].as<IntImmNode>()->value;
        int a1 = shape[1].as<IntImmNode>()->value;
        if (a0 != 1 && a1 != 1) {
          use_buffer = true;
        }
      } else if (kernel_layout == "OIHW4o") {
        int a2 = shape[2].as<IntImmNode>()->value;
        int a3 = shape[3].as<IntImmNode>()->value;
        if (a2 != 1 && a3 != 1) {
          use_buffer = true;
        }
      }
    }
    return use_buffer;
  }

  /*! \brief Temporary state for marking whether a visited function
   *         primitive supports texture storage scope */
  bool primitive_supports_texture_ = false;
  /*! \brief expr storage scope mapping for each output  */
  std::unordered_map<const ExprNode*, std::vector<std::string>> storage_scope_;
  /*! \brief output storage scopes used by consumers of expr key  */
  std::unordered_map<const ExprNode*, std::vector<std::string>> consumer_storage_scopes_;
  /*! \brief mapping of arguments to call to function variables*/
  std::unordered_map<Expr, std::vector<Var>, ObjectPtrHash, ObjectPtrEqual> args_to_vars_;
  /*! \brief mapping of arguments that can be converted to texture*/
  Map<Expr, Map<Expr, Array<String>>> accept_textures_;
  /*! \brief main attribute for expression*/
  tvm::Attrs expr_attrib;
  /*! \brief parameters that filter out from storage_map to use buffers*/
  std::unordered_set<Expr, ObjectPtrHash> buffers_params;
  /*! \brief arguments in expression that will use buffers*/
  std::unordered_set<Expr, ObjectPtrHash> buffers_args;
};

}  // namespace

/**
 * @brief rewrite of virtual devices, memory_scope part for expressions defined
 * by the StorageInfo analysis pass
 *
 * Currently this workflow supports analysis and rewriting of VirtualDevice for
 * Constants and function Variables
 */
class RewriteVDStorageScopes : public transform::DeviceAwareExprMutator {
  using VarMap = std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual>;

 public:
  using transform::DeviceAwareExprMutator::VisitExpr_;

  explicit RewriteVDStorageScopes(const Map<Expr, Map<Expr, Array<String>>>& storage_scope)
      : transform::DeviceAwareExprMutator(Optional<IRModule>()), storage_scope_(storage_scope) {}

  Function Rewrite(const Expr& expr) { return Downcast<Function>(Mutate(expr)); }

  Expr VisitExpr_(const VarNode* vn) final {
    if (storage_scope_.find(GetRef<Expr>(vn)) != storage_scope_.end() &&
        storage_scope_[GetRef<Expr>(vn)].find(Expr()) != storage_scope_[GetRef<Expr>(vn)].end() &&
        storage_scope_[GetRef<Expr>(vn)][Expr()][0] != "global") {
      Var c = Var(vn->vid, vn->type_annotation, vn->span);
      auto virtual_device = GetVirtualDevice(GetRef<Expr>(vn));
      c->virtual_device_ =
          VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                        virtual_device->target, storage_scope_[GetRef<Expr>(vn)][Expr()][0]);
      return std::move(c);
    }
    return GetRef<Var>(vn);
  }

  Expr VisitExpr_(const ConstantNode* vn) final {
    if (storage_scope_.find(GetRef<Expr>(vn)) != storage_scope_.end() &&
        storage_scope_[GetRef<Expr>(vn)].find(Expr()) != storage_scope_[GetRef<Expr>(vn)].end()) {
      Expr c = Constant(vn->data, vn->span);
      auto virtual_device = GetVirtualDevice(GetRef<Expr>(vn));
      c = OnDevice(
          c,
          VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                        virtual_device->target, storage_scope_[GetRef<Expr>(vn)][Expr()][0]),
          true);
      return c;
    }
    return GetRef<Constant>(vn);
  }

  Expr DeviceAwareVisitExpr_(const CallNode* call_node) final {
    // we need to duplicate ExprMutator::VisitExpr_ to correct argument scopes and
    // put device_copy
    auto new_op = this->Mutate(call_node->op);

    tvm::Array<Type> ty_args;
    ty_args.reserve(call_node->type_args.size());

    for (auto ty_arg : call_node->type_args) {
      auto new_ty_arg = this->VisitType(ty_arg);
      ty_args.push_back(new_ty_arg);
    }

    tvm::Array<Expr> call_args;
    call_args.reserve(call_node->args.size());
    for (auto arg : call_node->args) {
      auto new_arg = this->Mutate(arg);
      // verification if we need to put device_copy
      if (storage_scope_.count(arg) && storage_scope_[arg].count(GetRef<Expr>(call_node))) {
        auto virtual_device = GetVirtualDevice(GetRef<Expr>(call_node));
        VirtualDevice virtual_device_from =
            VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                          virtual_device->target, virtual_device->memory_scope);
        VirtualDevice virtual_device_to =
            VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                          virtual_device->target, storage_scope_[arg][GetRef<Expr>(call_node)][0]);
        new_arg = DeviceCopy(new_arg, virtual_device_from, virtual_device_to);
        new_arg = OnDevice(
            new_arg,
            VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                          virtual_device->target, storage_scope_[arg][GetRef<Expr>(call_node)][0]),
            true);
      }
      call_args.push_back(new_arg);
    }

    auto new_call = WithFields(GetRef<Call>(call_node), new_op, call_args, {}, ty_args);

    auto virtual_device = GetVirtualDevice(GetRef<Expr>(call_node));
    std::string memory_scope = "";
    if (storage_scope_.find(GetRef<Expr>(call_node)) != storage_scope_.end() &&
        storage_scope_[GetRef<Expr>(call_node)].find(Expr()) !=
            storage_scope_[GetRef<Expr>(call_node)].end()) {
      memory_scope = storage_scope_[GetRef<Expr>(call_node)][Expr()][0];
    } else if (virtual_device->memory_scope != "") {
      memory_scope = virtual_device->memory_scope;
    } else if (!call_node->op.as<FunctionNode>()) {
      memory_scope = "";
    }
    if (!memory_scope.empty()) {
      new_call =
          OnDevice(new_call,
                   VirtualDevice(virtual_device->device_type(), virtual_device->virtual_device_id,
                                 virtual_device->target, memory_scope),
                   true);
    }
    return std::move(new_call);
  }

 private:
  Map<Expr, Map<Expr, Array<String>>> storage_scope_;
  VarMap new_vars_;
  Array<String> current_function_scope_;
};

Map<Expr, Map<Expr, Array<String>>> CollectTextureStorage(const Expr& expr) {
  return StorageInfo::GetStorageMap(expr);
}

/**
 * @brief Collects all target devices participated in graph
 */
class CollectVirtualDevices : public transform::DeviceAwareExprVisitor {
 public:
  CollectVirtualDevices() : transform::DeviceAwareExprVisitor(Optional<IRModule>()) {}
  /**
   * @brief Get all unique device elements from target of each VirtualDevice
   *
   * @param expr - IR
   * @return set of devices
   */
  std::set<std::string> GetDevices(const Expr& expr) {
    this->Run(expr);
    return std::move(devices_);
  }

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

  void DeviceAwareVisitExpr_(const CallNode* call) final {
    auto vd = GetVirtualDevice(GetRef<Expr>(call));
    if (vd != VirtualDevice::FullyUnconstrained()) {
      if (Optional<String> t_device = vd->target->GetAttr<String>("device")) {
        devices_.insert(vd->target->kind->name + "." + t_device.value());
      }
    }
    for (auto& arg : call->args) {
      Visit(arg);
    }
  }

  void Run(const Expr& expr) { VisitExpr(expr); }
  using transform::DeviceAwareExprVisitor::VisitExpr_;
  std::set<std::string> devices_;
};

/*!
 * \brief Collect the target specific tensor storage info for each expression's output.
 * \param expr The expression.
 * \return The device based storage mapping.
 */
Map<Expr, Map<Expr, Array<String>>> CollectStorageInfo(const Expr& expr) {
  std::set<std::string> device_types = CollectVirtualDevices().GetDevices(expr);
  // TODO(amalyshe): current approach collects all targets withing graph and call the only
  // function corresponding to all these targets in alphabetic order
  // this will work reliable only for case of only one device and should be redesigned
  // to handle common case
  std::string ftarget_prefix = "relay.backend";
  for (auto& dev_id : device_types) {
    ftarget_prefix += (std::string(".") + dev_id);
  }

  Map<Expr, Map<Expr, Array<String>>> storage_info = {};
  if (const auto* f = runtime::Registry::Get(ftarget_prefix + "._CollectStorageInfo")) {
    storage_info = (*f)(expr);
  }
  return storage_info;
}

Expr AnnotateMemoryScopeExpr(const Expr& expr, const IRModule& mod) {
  auto storage_scope = CollectStorageInfo(expr);
  if (storage_scope.size()) {
    return RewriteVDStorageScopes(storage_scope).Rewrite(expr);
  } else {
    return expr;
  }
}

namespace transform {
tvm::transform::Pass AnnotateMemoryScope() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(AnnotateMemoryScopeExpr(f, m));
      };
  return CreateFunctionPass(pass_func, 2, "AnnotateMemoryScope", {});
}
}  // namespace transform

TVM_REGISTER_GLOBAL("relay.backend.opencl.adreno._CollectStorageInfo")
    .set_body_typed(CollectTextureStorage);

}  // namespace relay
}  // namespace tvm
