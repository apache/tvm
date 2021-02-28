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
    void BackwardPropagateConsumerScope(const ExprNode* expr) {
      auto consumer_scopes_it = consumer_storage_scopes_.find(expr);
      if (consumer_scopes_it != consumer_storage_scopes_.end())
      {
        bool all_consumers_support_texture = true;
        for (auto& consumer_scope : consumer_scopes_it->second) {
          if (consumer_scope != "texture") {
            all_consumers_support_texture = false;
          }
        }
        if (all_consumers_support_texture)
        {
          storage_scope_[expr] = "texture";
        }
      }
    }

    void VisitExpr_(const ConstantNode* cn) final {
      BackwardPropagateConsumerScope(cn);
    }

    void VisitExpr_(const CallNode* call) final {
      // Check the contents of this primitive function
      if (const auto* fn = call->op.as<FunctionNode>()) {
        if (fn->HasNonzeroAttr(attr::kPrimitive)) {
          primitive_supports_texture_ = false;
          Visit(call->op);
          if (primitive_supports_texture_)
          {
            storage_scope_[call] = "texture";
            for (auto& arg : call->args) {
              consumer_storage_scopes_[arg.get()].push_back("texture");
            }
          }
        }
      }

      if (call->attrs.as<Conv2DAttrs>()) {
        primitive_supports_texture_ = true;
      }
      for (auto& arg : call->args) {
        Visit(arg);
      }
    }

    void VisitExpr_(const VarNode* vn) final {
      BackwardPropagateConsumerScope(vn);
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

TVM_REGISTER_GLOBAL("relay.analysis.CollectStorageInfo").set_body_typed(CollectStorageInfo);

}  // namespace relay
}  // namespace tvm
