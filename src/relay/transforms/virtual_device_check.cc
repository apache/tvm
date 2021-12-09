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

#include "./virtual_device_check.h"

#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

using tvm::relay::transform::CreateFunctionPass;
using tvm::transform::PassContext;

void DeviceChecker::VisitExpr(const Expr& e) {
    ExprVisitor::VisitExpr(e);
}

// TODO(@jroesch, @junru): we need to deal with unique spans for global/var.
void DeviceChecker::VisitExpr_(const VarNode* op) {
    // ICHECK((op->virtual_device_.defined())) << "VarNode's virtual device should not be null";
}
void DeviceChecker::VisitExpr_(const GlobalVarNode* op) {}
void DeviceChecker::VisitExpr_(const ConstantNode* op) {}

void DeviceChecker::VisitExpr_(const TupleNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const FunctionNode* op) { 
    ICHECK(!op->virtual_device()->IsFullyConstrained());
    for (auto var : op->params) {
        ICHECK(!op->virtual_device()->IsFullyConstrained());
    }
    ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const CallNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const LetNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const IfNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const OpNode* op) {}

void DeviceChecker::VisitExpr_(const TupleGetItemNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const RefCreateNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const RefReadNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const RefWriteNode* op) { ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const ConstructorNode* op) {}  // ExprVisitor::VisitExpr_(op); }

void DeviceChecker::VisitExpr_(const MatchNode* op) { ExprVisitor::VisitExpr_(op); }


void DeviceChecker::VisitType(const Type& t) {}
void DeviceChecker::VisitClause(const Clause& c) {}
void DeviceChecker::VisitPattern(const Pattern& c) {}

tvm::transform::Pass VirtualDeviceCheck() {
  return CreateFunctionPass(
      [](const Function& func, const IRModule& mod, const PassContext& ctx) {
        ICHECK(ctx->diag_ctx) << "Diagnostic context must be set.";
        DeviceChecker checker;
        checker.VisitExpr(func);
        return func;
      },
      0, "VirtualDeviceCheck", {});
}

TVM_REGISTER_GLOBAL("VirtualDeviceCheck").set_body_typed([]() { return VirtualDeviceCheck(); });

} // namespace relay
} // namespace tvm