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
 * \file verify_memory.cc
 * \brief Pass to check if memory accesses are legal.
 */
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {
namespace {

/*!
 * \brief Verify if memory accesses are legal.
 *
 *  In the case that tgt is cuda, if workload is not bound with
 *  threads, CPU code is generated that tries to access GPU memory,
 *  which is illegal.
 *
 *  This pass performs such verification by checking if all
 *  memory accesses are bound with threads when device type is GPU.
 */
class MemoryAccessVerifier final : protected StmtExprVisitor {
 public:
  /// Special member functions
  //@{
  explicit MemoryAccessVerifier(PrimFunc f, int device_type) : func_(f), dev_type_(device_type) {}
  virtual ~MemoryAccessVerifier() = default;
  MemoryAccessVerifier(const MemoryAccessVerifier&) = delete;
  MemoryAccessVerifier(MemoryAccessVerifier&&) = delete;
  MemoryAccessVerifier& operator=(const MemoryAccessVerifier&) = delete;
  MemoryAccessVerifier& operator=(MemoryAccessVerifier&&) = delete;
  //@}

  /// Interface to perform memory access verification
  void Run() {
    if (!IsGPUDevice(dev_type_) && !IsFPGADevice(dev_type_)) return;
    StmtExprVisitor::VisitStmt(func_->body);
  }

  /// Verification result
  bool Failed() const { return failure_; }

 protected:
  /// Visitor implementation
  //@{
  void VisitExpr(const PrimExpr& n) final {
    if (Failed()) return;
    StmtExprVisitor::VisitExpr(n);
  }

  void VisitStmt(const Stmt& n) final {
    if (Failed()) return;
    StmtExprVisitor::VisitStmt(n);
  }

  void VisitStmt_(const LetStmtNode* op) final {
    // Book keep definitions
    defs_[op->var.get()] = op->value;
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (!InThreadEnv() &&
        (op->attr_key == attr::thread_extent || op->attr_key == attr::pipeline_exec_scope)) {
      EnterThreadEnv();
      StmtExprVisitor::VisitStmt_(op);
      ExitThreadEnv();
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const LoadNode* op) final {
    HandleLoadStoreToVariable(op->buffer_var);
    return StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const StoreNode* op) final {
    HandleLoadStoreToVariable(op->buffer_var);
    return StmtExprVisitor::VisitStmt_(op);
  }
  //@}

  /// Check if the value of a Variable comes from function argument.
  bool IsFromFunctionArgs(const VarNode* var) const {
    const VarNode* V = var;
    for (auto kv : func_->buffer_map) {
      if (V == kv.second->data.get()) return true;
    }

    while (true) {
      // Variable is from function args. Return true.
      if (V == func_->params[0].get()) return true;

      // The value is expected to come from a tvm_struct_get Call.
      // Get the first argument of tvm_struct_get, and continue.
      const auto& iter = defs_.find(V);
      if (iter == defs_.end()) return false;
      const CallNode* C = iter->second.as<const CallNode>();
      if (!C || !C->op.same_as(builtin::tvm_struct_get())) return false;
      V = C->args[0].as<VarNode>();
    }
    return false;
  }

  /// Handle memory access to a Variable
  void HandleLoadStoreToVariable(const Var& var) {
    // We skip the access within thread env.
    if (InThreadEnv()) return;

    // We only handle the variable from function argument.
    // If it does not come from args, then it could be allocated internally,
    // it may possibly be in host or device address space.
    // We do not handle this case, and skip it conservatively.
    if (!IsFromFunctionArgs(var.get())) return;

    // The verification fails in this case.
    SetFailure();
  }

  /// Status getter/setter
  //@{
  bool InThreadEnv() const { return in_thread_env_; }
  void EnterThreadEnv() { in_thread_env_ = true; }
  void ExitThreadEnv() { in_thread_env_ = false; }
  void SetFailure() { failure_ = true; }
  //@}

  /// Check if a given DLDeviceType/TVMDeviceExtType value denotes GPU device.
  static bool IsGPUDevice(int dev_type) {
    return kDLGPU == dev_type || kDLOpenCL == dev_type || kDLVulkan == dev_type ||
           kDLMetal == dev_type || kDLROCM == dev_type || kOpenGL == dev_type;
  }
  /// Check if a given DLDeviceType/TVMDeviceExtType value denotes FPGA device.
  static bool IsFPGADevice(int dev_type) { return kDLSDAccel == dev_type || kDLAOCL == dev_type; }

 private:
  /// Status of visitor
  //@{
  bool in_thread_env_{false};
  bool failure_{false};  ///< If the verification fails (i.e. has illegal access)
  //@}
  tir::PrimFunc func_{nullptr};                        ///< Function to be verified.
  int dev_type_{kDLCPU};                               ///< Device type
  std::unordered_map<const VarNode*, PrimExpr> defs_;  ///< Variable definitions
};
}  // namespace

/// Interface of VerifyMemory pass
bool VerifyMemory(const PrimFunc& func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  CHECK(target.defined()) << "LowerWarpMemory: Require the target attribute";

  if (func->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) ==
      CallingConv::kDefault) {
    MemoryAccessVerifier v(func, target.value()->id->device_type);
    v.Run();
    return !v.Failed();
  } else {
    return true;
  }
}

TVM_REGISTER_GLOBAL("tir.analysis.verify_memory").set_body_typed(VerifyMemory);

namespace transform {

Pass VerifyMemory() {
  auto pass_func =
      [=](IRModule mod, PassContext ctx) {
        for (auto kv : mod->functions) {
          if (auto* n = kv.second.as<PrimFuncNode>()) {
            auto func = GetRef<PrimFunc>(n);
            CHECK(VerifyMemory(func))
                << "RuntimeError: Direct host side access to device memory is detected."
                << " Did you forget to bind?\n"
                << func;
          }
        }
        return mod;
      };
  return tvm::transform::CreateModulePass(pass_func, 0, "tir.VerifyMemory", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VerifyMemory").set_body_typed(VerifyMemory);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
