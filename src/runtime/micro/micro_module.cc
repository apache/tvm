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
   * \param binary name of the binary to be loaded
   */
  void InitMicroModule(const std::string binary) {
    session_ = MicroSession::Global();
    low_level_device_ = session_->low_level_device();
    binary_ = binary;
    LoadBinary();
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
  /*! \brief loaded module text start address */
  dev_base_offset text_start_;
  /*! \brief loaded module data start address */
  dev_base_offset data_start_;
  /*! \brief loaded module bss start address */
  dev_base_offset bss_start_;
  /*! \brief size of module text section */
  size_t text_size_;
  /*! \brief size of module data section */
  size_t data_size_;
  /*! \brief size of module bss section */
  size_t bss_size_;
  /*! \brief module binary */
  std::string binary_;
  /*! \brief global session pointer */
  std::shared_ptr<MicroSession> session_;
  /*! \brief low-level device pointer */
  std::shared_ptr<LowLevelDevice> low_level_device_;
  /*! \brief symbol map to addresses */
  SymbolMap symbol_map_;

  void PatchImplHole(const std::string func_name) {
    // std::cout << "func_name: " << func_name << std::endl;
    // std::cout << "base_addr: 0x" << std::hex << low_level_device_->base_addr().val_ << std::endl;
    // std::cout << "text_start: " << std::hex << "0x" << text_start_.val_ << std::endl;
    const dev_base_offset init_impl_offset = session_->init_symbol_map()[func_name];
    // std::cout << "init_impl_offset: 0x" << std::hex << init_impl_offset.val_ << std::endl;
    void* init_impl_addr = (void*) (low_level_device_->base_addr().val_ + init_impl_offset.val_);
    // std::cout << "init_impl_addr: 0x" << std::hex << init_impl_addr << std::endl;
    std::stringstream func_name_underscore;
    func_name_underscore << func_name << "_";
    const dev_base_offset lib_hole_offset = symbol_map_[func_name_underscore.str()];
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
    // TODO(weberlo): Move the patching below to the init stub.
    dev_base_offset workspace_start_hole_offset = session_->init_symbol_map()["workspace_start"];
    dev_base_offset workspace_curr_hole_offset = session_->init_symbol_map()["workspace_curr"];
    void* workspace_hole_fill = (void*) (kWorkspaceStart.val_ + low_level_device_->base_addr().val_);

    // session_->low_level_device()->Read(workspace_start_hole_offset, &tmp, sizeof(void*));
    // std::cout << "workspace start addr (before): 0x" << std::hex << tmp << std::endl;
    session_->low_level_device()->Write(workspace_start_hole_offset, &workspace_hole_fill, sizeof(void*));
    // session_->low_level_device()->Read(workspace_start_hole_offset, &tmp, sizeof(void*));
    // std::cout << "workspace start addr (after): 0x" << std::hex << tmp << std::endl;

    // session_->low_level_device()->Read(workspace_curr_hole_offset, &tmp, sizeof(void*));
    // std::cout << "workspace curr addr (before): 0x" << std::hex << tmp << std::endl;
    session_->low_level_device()->Write(workspace_curr_hole_offset, &workspace_hole_fill, sizeof(void*));
    // session_->low_level_device()->Read(workspace_curr_hole_offset, &tmp, sizeof(void*));
    // std::cout << "workspace curr addr (after): 0x" << std::hex << tmp << std::endl;
  }

  void LoadBinary() {
    text_size_ = GetSectionSize(binary_, kText);
    data_size_ = GetSectionSize(binary_, kData);
    bss_size_ = GetSectionSize(binary_, kBss);

    text_start_ = session_->AllocateInSection(kText, text_size_);
    data_start_ = session_->AllocateInSection(kData, data_size_);
    bss_start_ = session_->AllocateInSection(kBss, bss_size_);
    CHECK(text_start_.val_ != 0 && data_start_.val_ != 0 && bss_start_.val_ != 0)
      << "Not enough space to load module on device";
    const dev_base_addr base_addr = low_level_device_->base_addr();
    std::string relocated_bin = RelocateBinarySections(
        binary_,
        (void*) GetAddr(text_start_, base_addr).val_,
        (void*) GetAddr(data_start_, base_addr).val_,
        (void*) GetAddr(bss_start_, base_addr).val_);
    std::string text_contents = ReadSection(relocated_bin, kText);
    std::string data_contents = ReadSection(relocated_bin, kData);
    std::string bss_contents = ReadSection(relocated_bin, kBss);
    low_level_device_->Write(text_start_, &text_contents[0], text_size_);
    low_level_device_->Write(data_start_, &data_contents[0], data_size_);
    low_level_device_->Write(bss_start_, &bss_contents[0], bss_size_);
    symbol_map_ = SymbolMap(relocated_bin, base_addr);

    // Patch device lib pointers.
    PatchImplHole("TVMBackendAllocWorkspace");
    PatchImplHole("TVMBackendFreeWorkspace");
    PatchImplHole("TVMAPISetLastError");
    /*
    std::cout << "alloc: " << GetSymbol(session_->init_symbol_map(), "TVMBackendAllocWorkspace", nullptr) << std::endl;
    std::cout << "free: " << GetSymbol(session_->init_symbol_map(), "TVMBackendFreeWorkspace", nullptr) << std::endl;
    std::cout << "error: " << GetSymbol(session_->init_symbol_map(), "TVMAPISetLastError", nullptr) << std::endl;
    std::cout << "alloc_hole_: " << GetSymbol(symbol_map_, "TVMBackendAllocWorkspace_", nullptr) << std::endl;
    std::cout << "free_hole_: " << GetSymbol(symbol_map_, "TVMBackendFreeWorkspace_", nullptr) << std::endl;
    std::cout << "error_hole_: " << GetSymbol(symbol_map_, "TVMAPISetLastError_", nullptr) << std::endl;
    std::cout << "alloc_hole: " << GetSymbol(symbol_map_, "TVMBackendAllocWorkspace", nullptr) << std::endl;
    std::cout << "free_hole: " << GetSymbol(symbol_map_, "TVMBackendFreeWorkspace", nullptr) << std::endl;
    std::cout << "error_hole: " << GetSymbol(symbol_map_, "TVMAPISetLastError", nullptr) << std::endl;
    */
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
    // no return value yet, but may implement in the future
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
  dev_base_offset func_offset = symbol_map_[name];
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
