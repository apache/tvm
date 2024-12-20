/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file inject_fence_proxy.cc
 * \brief Inject fence between generic and async proxies (sm90+)
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

enum class Proxy { kGeneric, kAsync, kBoth };

class ProxyMarker : public StmtVisitor {
 public:
  ProxyMarker() = default;

  Proxy GetProxy(const StmtNode* stmt) const {
    auto it = map_.find(stmt);
    // ICHECK(it != map_.end());
    // TODO: This is a hack implementation to avoid the ICHECK failure.
    if (it == map_.end()) {
      return Proxy::kGeneric;
    }
    return it->second;
  }

  Proxy GetProxy(const Stmt& stmt) const { return GetProxy(stmt.get()); }

  void VisitStmt_(const EvaluateNode* op) final {
    Proxy proxy = Proxy::kAsync;
    if (auto call = op->value.as<CallNode>()) {
      if (call->op.same_as(LDMatrixOp()) || call->op.same_as(STMatrixOp())) {
        proxy = Proxy::kGeneric;
      }
    }
    SetProxy(op, proxy);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    Proxy proxy = Proxy::kGeneric;
    SetProxy(op, proxy);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetProxy(op->seq[0]);
    for (auto stmt : op->seq) {
      if (role != GetProxy(stmt)) {
        role = Proxy::kBoth;
        break;
      }
    }
    SetProxy(op, role);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto role = GetProxy(op->then_case);
    if (op->else_case.defined()) {
      auto role_else = GetProxy(op->else_case.value());
      if (role != role_else) role = Proxy::kBoth;
    }
    SetProxy(op, role);
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    StmtVisitor::VisitStmt_(op);
    SetProxy(op, GetProxy(op->block));
  }

  template <class NodeType>
  void HandleBodyStmt(const NodeType* op) {
    StmtVisitor::VisitStmt_(op);
    SetProxy(op, GetProxy(op->body));
  }

  void VisitStmt_(const ForNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const LetStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AttrStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const AssertStmtNode* op) final { HandleBodyStmt(op); }
  void VisitStmt_(const BlockNode* op) final { HandleBodyStmt(op); }



 private:
  void SetProxy(const StmtNode* stmt, Proxy proxy) { map_[stmt] = proxy; }
  std::unordered_map<const StmtNode*, Proxy> map_;
};


class InjectFenceProxy : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    auto T = InjectFenceProxy();
    f.CopyOnWrite()->body = T(f->body);
    return f;
  }

 private:
  Proxy get_generic_proxy(const Stmt& stmt) {
    auto marker = ProxyMarker();
    marker(stmt);
    return marker.GetProxy(stmt);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    ICHECK(op->seq.size() > 0);
    Array<Stmt> new_body;
    Proxy cur_proxy, prev_proxy;
    auto fence_stmt = Evaluate(Call(DataType::Handle(), FenceProxyAsyncOp(), {}));
    prev_proxy = get_generic_proxy(op->seq[0]);
    new_body.push_back(VisitStmt(op->seq[0]));
    if (op->seq.size() > 1) {
      for (int i = 1; i < static_cast<int>(op->seq.size()); i++) {
        cur_proxy = get_generic_proxy(op->seq[i]);
        if (cur_proxy == Proxy::kAsync && prev_proxy == Proxy::kGeneric) {
          new_body.push_back(fence_stmt);
        }
        new_body.push_back(VisitStmt(op->seq[i]));
        prev_proxy = cur_proxy;
      }
    }
    ICHECK(new_body.size() > 0);
    return new_body.size() == 1 ? new_body[0] : SeqStmt(std::move(new_body));
  }

  // Stmt VisitStmt_(const ForNode* op) final {
  //   std::cout << "ForNode:" << op->body->GetTypeKey() << std::endl;
  //   return StmtExprMutator::VisitStmt_(op);
  // }

  InjectFenceProxy() = default;
};

using namespace tir::transform;

tvm::transform::Pass InjectFenceProxy() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return InjectFenceProxy::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectFenceProxy", {});
}

TVM_REGISTER_GLOBAL("tl.InjectFenceProxy").set_body_typed(InjectFenceProxy);

}  // namespace tl
}  // namespace tvm
