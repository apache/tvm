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
 * \file deivce_annotation.cc
 * \brief
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

class StorageInfo {
 public:
  static Map<Expr, String> GetStorageMap(const Expr& expr) {
    StorageInfo storage_info;
    storage_info.pre_visitor_ = PreDfsOrderVisitor();
    storage_info.pre_visitor_.Visit(expr);
    // TODO(csullivan): A unit test for legalization
    storage_info.pre_visitor_.LegalizeProducerStorage();
    for (auto& it : storage_info.pre_visitor_.storage_scope_) {
      storage_info.storage_map_.Set(GetRef<Expr>(it.first), String(it.second));
    }
    return storage_info.storage_map_;
  }

 private:
  class PreDfsOrderVisitor : private ExprVisitor {
   public:
    void Visit(const Expr& expr) {
      if (const auto* fn = expr.as<FunctionNode>()) {
        this->VisitExpr(fn->body);
        for (const auto& param : fn->params) {
          this->VisitExpr(param);
        }
      } else {
        this->VisitExpr(expr);
      }
    }

   private:
    std::string GetConsumerScope(const std::vector<std::string>& consumer_scopes) const {
      if (!consumer_scopes.size()) { return "global"; }
      std::string ref_scope = consumer_scopes.front();
      for (auto& consumer_scope : consumer_scopes) {
        if (consumer_scope != ref_scope) {
          return "global";
        }
      }
      return ref_scope;
    }

    void BackwardPropagateConsumerScope(const ExprNode* expr, std::string scope_suffix = "") {
      auto consumer_scopes_it = consumer_storage_scopes_.find(expr);
      if (consumer_scopes_it != consumer_storage_scopes_.end())
      {
        storage_scope_[expr] = GetConsumerScope(consumer_scopes_it->second);
        if (storage_scope_[expr] == "texture")
        {
          if (!scope_suffix.empty()) {
            storage_scope_[expr] += (":" + scope_suffix);
          }
        }
      }
    }

    void VisitExpr_(const ConstantNode* cn) final {
      BackwardPropagateConsumerScope(cn, "weight");
    }

    void VisitExpr_(const CallNode* call) final {
      // Check the contents of this primitive function
      if (const auto* fn = call->op.as<FunctionNode>()) {
        if (fn->HasNonzeroAttr(attr::kPrimitive)) {
          primitive_supports_texture_ = false;
          Visit(call->op);
          if (primitive_supports_texture_) {
            storage_scope_[call] = "texture";
          }
          for (auto& arg : call->args) {
            std::string scope = storage_scope_.count(call) ? storage_scope_[call] : "global";
            consumer_storage_scopes_[arg.get()].push_back(scope);
          }
        }
      }

      if (auto attrs = call->attrs.as<Conv2DAttrs>()) {
        if (attrs->data_layout == "NCHW4c" && attrs->kernel_layout == "OIHW4o") {
          primitive_supports_texture_ = true;
        }
      }
      for (auto& arg : call->args) {
        Visit(arg);
      }
    }

    void VisitExpr_(const VarNode* vn) final {
      BackwardPropagateConsumerScope(vn);
    }

    void LegalizeProducerStorage() {
      for (auto& kv : consumer_storage_scopes_) {
        const ExprNode* producer = kv.first;
        std::string legal_scope = GetConsumerScope(kv.second);
        if (storage_scope_.count(producer)) {
          if (storage_scope_[producer].find(legal_scope) == std::string::npos) {
            storage_scope_[producer] = legal_scope;
          }
        }
      }
    }

    bool primitive_supports_texture_ = false;
    std::unordered_map<const ExprNode*, std::string> storage_scope_;
    std::unordered_map<const ExprNode*, std::vector<std::string>> consumer_storage_scopes_;
    friend StorageInfo;
  };

  PreDfsOrderVisitor pre_visitor_;
  Map<Expr, String> storage_map_;
};

Map<Expr, String> CollectStorageInfo(const Expr& expr) { return StorageInfo::GetStorageMap(expr); }

namespace {
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

TVM_REGISTER_GLOBAL("relay.analysis.CollectStorageInfo").set_body_typed(CollectStorageInfo);

TVM_REGISTER_GLOBAL("relay.backend.opencl.adreno._CollectBufferBinds").set_body_typed(CollectBufferBinds);

}  // namespace relay
}  // namespace tvm
