/*!
*  Copyright (c) 2019 by Contributors
* \file micro_module.cc
*/

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <unordered_map>
#include <string>
#include "micro_session.h"
#include "low_level_device.h"
#include "micro_common.h"
#include "../pack_args.h"

namespace tvm {
namespace runtime {
/*!
 * \brief module for uTVM micro devices
 */
class MicroModuleNode final : public ModuleNode {
 public:
  MicroModuleNode() {}

  ~MicroModuleNode() {}

  const char* type_key() const final {
    return "micro";
  }

  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  /*!
   * \brief initializes module by establishing device connection and loads binary
   * \param binary_path path of the binary to be loaded
   */
  void InitMicroModule(const std::string binary_path) {
    session_ = MicroSession::Global();
    low_level_device_ = session_->low_level_device();
    binary_path_ = binary_path;
    binary_info_ = session_->LoadBinary(binary_path_);
    // Patch device lib pointers.
    PatchImplHole("TVMBackendAllocWorkspace");
    PatchImplHole("TVMBackendFreeWorkspace");
    PatchImplHole("TVMAPISetLastError");
  }

  /*!
   * \brief runs selected function on the micro device
   * \param func name of the function to be run
   * \param func_addr address of the function to be run
   * \param args type-erased arguments passed to the function
   */
  void RunFunction(std::string func, dev_base_offset func_offset, TVMArgs args) {
    session_->PushToExecQueue(func_offset, args);
  }

 private:
  /*! \brief module binary info */
  BinaryInfo binary_info_;
  /*! \brief path to module binary */
  std::string binary_path_;
  /*! \brief global session pointer */
  std::shared_ptr<MicroSession> session_;
  /*! \brief low-level device pointer */
  std::shared_ptr<LowLevelDevice> low_level_device_;

  SymbolMap symbol_map() {
    return binary_info_.symbol_map;
  }

  void PatchImplHole(const std::string func_name) {
    // std::cout << "func_name: " << func_name << std::endl;
    // std::cout << "base_addr: 0x" << std::hex << low_level_device_->base_addr().val_ << std::endl;
    // std::cout << "text_start: " << std::hex << "0x" << binary_info_.text.start.val_ << std::endl;
    const dev_base_offset init_impl_offset = session_->init_symbol_map()[func_name];
    // std::cout << "init_impl_offset: 0x" << std::hex << init_impl_offset.val_ << std::endl;
    void* init_impl_addr = (void*) (low_level_device_->base_addr().val_ + init_impl_offset.val_);
    // std::cout << "init_impl_addr: 0x" << std::hex << init_impl_addr << std::endl;
    std::stringstream func_name_underscore;
    func_name_underscore << func_name << "_";
    const dev_base_offset lib_hole_offset = symbol_map()[func_name_underscore.str()];
    // std::cout << "lib_hole_offset: 0x" << std::hex << lib_hole_offset.val_ << std::endl;
    // std::cout << "lib_hole_addr: 0x" << std::hex << (low_level_device_->base_addr().val_ + lib_hole_offset.val_) << std::endl;
    // void* tmp;
    // session_->low_level_device()->Read(lib_hole_offset, &tmp, sizeof(void*));
    // std::cout << "tmp addr (before): 0x" << std::hex << tmp << std::endl;
    session_->low_level_device()->Write(lib_hole_offset, &init_impl_addr, sizeof(void*));
    // session_->low_level_device()->Read(lib_hole_offset, &tmp, sizeof(void*));
    // std::cout << "tmp addr: 0x" << std::hex << tmp << std::endl;
    // std::cout << "tmp offset: 0x" << std::hex << (((uintptr_t) tmp) - low_level_device_->base_addr().val_) << std::endl;
    // std::cout << std::endl;
  }
};

class MicroWrappedFunc {
 public:
  MicroWrappedFunc(MicroModuleNode* m,
                   const std::string& func_name,
                   dev_base_offset func_offset) {
    m_ = m;
    func_name_ = func_name;
    func_offset_ = func_offset;
  }

  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    // TODO(weberlo): no return value yet, but may implement in the future
    m_->RunFunction(func_name_, func_offset_, args);
  }

 private:
  // internal module
  MicroModuleNode* m_;
  // name of the function
  std::string func_name_;
  // address of the function to be called
  dev_base_offset func_offset_;
};

PackedFunc MicroModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  dev_base_offset func_offset = symbol_map()[name];
  MicroWrappedFunc f(this, name, func_offset);
  return PackFuncVoidAddr(f, std::vector<TVMType>());
}

// register loadfile function to load module from Python frontend
TVM_REGISTER_GLOBAL("module.loadfile_micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroModuleNode> n = std::make_shared<MicroModuleNode>();
    n->InitMicroModule(args[0]);
    *rv = runtime::Module(n);
    });
}  // namespace runtime
}  // namespace tvm
