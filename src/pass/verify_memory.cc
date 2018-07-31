/*!
 *  Copyright (c) 2018 by Contributors
 * \file verify_memory.cc
 * \brief Pass to check if memory accesses are legal.
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {
namespace {

/*!
 * \brief Verify if memory accesses are legal.
 *
 *  In the case that tgt is cuda, if workload is not bound with
 *  threads, CPU code is generated that tries to access GPU memory,
 *  which is illegal.
 *
 *  This pass performs such verification by checking if all Producer/Consumer
 *  with memory accesses are bound with threads when device type is GPU.
 */
class MemoryAccessVerifier final : protected IRVisitor {
 public:
  /// Special member functions
  //@{
  explicit MemoryAccessVerifier(LoweredFunc f, int device_type)
      : func_(f), dev_type_(device_type) {}
  virtual ~MemoryAccessVerifier() = default;
  MemoryAccessVerifier(const MemoryAccessVerifier &) = delete;
  MemoryAccessVerifier(MemoryAccessVerifier &&) = delete;
  MemoryAccessVerifier &operator=(const MemoryAccessVerifier &) = delete;
  MemoryAccessVerifier &operator=(MemoryAccessVerifier &&) = delete;
  //@}

  /// Interface to perform memory access verification
  void Run() {
    if (!IsGPUDevice(dev_type_) && !IsFPGADevice(dev_type_)) return;
    IRVisitor::Visit(func_->body);
  }

  /// Verification result
  bool Failed() const { return failure_; }

 protected:
  /// Visitor implementation
  //@{
  void Visit(const NodeRef &n) final {
    if (Failed()) return;
    IRVisitor::Visit(n);
  }

  void Visit_(const LetStmt *op) final {
    // Book keep definitions
    defs_[op->var.get()] = op->value;
    return IRVisitor::Visit_(op);
  }

  void Visit_(const AttrStmt *op) final {
    if (!InThreadEnv() && (op->attr_key == attr::thread_extent ||
                           op->attr_key == attr::pipeline_exec_scope)) {
      EnterThreadEnv();
      IRVisitor::Visit_(op);
      ExitThreadEnv();
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const ProducerConsumer *op) final {
    EnterProducerConsumer(op);
    IRVisitor::Visit_(op);
    ExitProducerConsumer();
  }

  void Visit_(const Load *op) final {
    HandleLoadStoreToVariable(op->buffer_var);
    return IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) final {
    HandleLoadStoreToVariable(op->buffer_var);
    return IRVisitor::Visit_(op);
  }
  //@}

  /// Check if the value of a Variable comes from function argument.
  bool IsFromFunctionArgs(const Variable *var) const {
    const Variable *V = var;
    while (true) {
      CHECK(V) << "Invalid Variable\n";

      // Variable is from function args. Return true.
      if (V == func_->args[0].node_.get()) return true;

      // The value is expected to come from a tvm_struct_get Call.
      // Get the first argument of tvm_struct_get, and continue.
      const auto &iter = defs_.find(V);
      if (iter == defs_.end()) return false;
      const Call *C = iter->second.as<const Call>();
      if (!C || C->name != intrinsic::tvm_struct_get) return false;
      V = C->args[0].as<Variable>();
    }
    return false;
  }

  /// Handle memory access to a Variable
  void HandleLoadStoreToVariable(const VarExpr &var) {
    // We skip the access within thread env.
    if (InThreadEnv()) return;

    // We only check access within a producer/consumer.
    // Because for load/store out side of producer/consumer,
    // they don't have to be in thread env to stay legal (e.g. Load of args).
    if (!InProducerConsumer()) return;

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
  bool InProducerConsumer() const { return pc_ != nullptr; }
  const ProducerConsumer *GetCurrentProducerConsumer() const { return pc_; }
  void EnterProducerConsumer(const ProducerConsumer *pc) { this->pc_ = pc; }
  void ExitProducerConsumer() { pc_ = nullptr; }
  void SetFailure() { failure_ = true; }
  //@}

  /// Check if a given DLDeviceType/TVMDeviceExtType value denotes GPU device.
  static bool IsGPUDevice(int dev_type) {
    return kDLGPU == dev_type || kDLOpenCL == dev_type ||
           kDLVulkan == dev_type || kDLMetal == dev_type ||
           kDLROCM == dev_type || kOpenGL == dev_type;
  }
  /// Check if a given DLDeviceType/TVMDeviceExtType value denotes FPGA device.
  static bool IsFPGADevice(int dev_type) {
    return kDLSDAccel == dev_type || kDLAOCL == dev_type;
  }

 private:
  /// Status of visitor
  //@{
  bool in_thread_env_{false};
  const ProducerConsumer *pc_{nullptr};
  bool failure_{false};  ///< If the verification fails (i.e. has illegal access)
  //@}
  LoweredFunc func_{nullptr};  ///< Function to be verified.
  int dev_type_{kDLCPU};       ///< Device type
  std::unordered_map<const Variable *, Expr> defs_;  ///< Variable definitions
};
}  // namespace

/// Interface of VerifyMemory pass
bool VerifyMemory(LoweredFunc func, int device_type) {
  MemoryAccessVerifier v(func, device_type);
  v.Run();
  return !v.Failed();
}

}  // namespace ir
}  // namespace tvm
