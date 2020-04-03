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
 * \file bind_device_type.cc
 * \brief Bind the device type according to the target field.
 */
#include <tvm/ir/transform.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/analysis.h>
#include <tvm/target/target.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tir {

class DeviceTypeBinder: public StmtExprMutator {
 public:
  explicit DeviceTypeBinder(int device_type)
      : device_type_(device_type) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::device_context_type) {
      if (const VarNode* var = op->value.as<VarNode>()) {
        var_ = var;
        PrimExpr value = make_const(op->value.dtype(), device_type_);
        Stmt body = StmtExprMutator::VisitStmt_(op);
        var_ = nullptr;
        std::ostringstream os;
        os << "device_type need to be " << device_type_;
        return AssertStmtNode::make(op->value == value, os.str(), body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    // eager simplify if guard.
    Stmt res = StmtExprMutator::VisitStmt_(op);
    op = res.as<IfThenElseNode>();
    if (is_zero(op->condition)) {
      if (op->else_case.defined()) return op->else_case;
      return EvaluateNode::make(0);
    }
    if (is_one(op->condition)) {
      return op->then_case;
    }
    return res;
  }

  PrimExpr VisitExpr_(const NENode* op) final {
    // eager check NE for device check
    PrimExpr res = StmtExprMutator::VisitExpr_(op);
    op = res.as<NENode>();
    if (tir::ExprDeepEqual()(op->a, op->b)) {
      return make_const(op->dtype, false);
    }
    return res;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (op == var_) {
      return make_const(op->dtype, device_type_);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

 public:
  const VarNode* var_{nullptr};
  int device_type_;
};

namespace transform {

Pass BindDeviceType() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    CHECK(target.defined())
        << "BindDeviceType: Require the target attribute";
    n->body = DeviceTypeBinder(target->device_type)(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BindDeviceType", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BindDeviceType")
.set_body_typed(BindDeviceType);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
