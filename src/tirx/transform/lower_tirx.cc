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
 * \file lower_tirx.cc
 * \brief Compose the TIRx lowering pipeline from individual passes.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {
namespace transform {

namespace {

/*!
 * \brief Strip ExecScopeStmt wrappers from lowered TIRX output.
 *
 * ExecScopeStmt is required while lowering TIRX ops and resolving scope IDs/slices.
 * After those passes finish, the wrappers are no longer needed and should not be
 * present in the final LowerTIRx output.
 */
class ExecScopeStripper : public StmtExprMutator {
 public:
  static Stmt Strip(const Stmt& stmt) { return ExecScopeStripper()(stmt); }

 private:
  Stmt VisitStmt_(const ExecScopeStmtNode* op) final { return VisitStmt(op->body); }
};

Pass LowerTIRxStripExecScope() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = ExecScopeStripper::Strip(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxStripExecScope", {});
}

}  // namespace

Pass LowerTIRx() {
  std::vector<tvm::transform::Pass> passes = {TilePrimitiveDispatch()};
  if (std::getenv("TVM_PRINT_AFTER_TIRX_DISPATCH_OPS")) {
    passes.push_back(tvm::transform::PrintIR());
  }
  passes.push_back(LowerTIRxCleanup());
  passes.push_back(LowerTIRxStripExecScope());
  return tvm::transform::Sequential(passes, "tirx.LowerTIRx");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.transform.TilePrimitiveDispatch", TilePrimitiveDispatch)
      .def("tirx.transform.LowerTIRxCleanup", LowerTIRxCleanup)
      .def("tirx.transform.LowerTIRx", LowerTIRx);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
