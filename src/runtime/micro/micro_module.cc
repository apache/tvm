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
    // TODO: if first MicroModule, then load init section in MicroSession
    // this will be handled by micro_init that loads MicroSession
    session_ = MicroSession::Global();
    lldevice_ = session_->low_level_device();
    binary_ = binary;
    LoadBinary();
  }

  /*!
   * \brief runs selected function on the micro device
   * \param func name of the function to be run
   * \param func_addr address of the function to be run
   * \param args type-erased arguments passed to the function
   */
  void RunFunction(std::string func, void* func_addr, TVMArgs args) {
    session_->PushToExecQueue(func_addr, args);
  }

 private:
  /*! \brief loaded module text start address */
  void* text_start_;
  /*! \brief loaded module data start address */
  void* data_start_;
  /*! \brief loaded module bss start address */
  void* bss_start_;
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
  std::shared_ptr<LowLevelDevice> lldevice_;
  /*! \brief symbol map to addresses */
  std::unordered_map<std::string, void*> symbol_map_;

  void LoadBinary() {
    text_size_ = GetSectionSize(binary_, kText);
    data_size_ = GetSectionSize(binary_, kData);
    bss_size_ = GetSectionSize(binary_, kBss);
    text_start_ = session_->AllocateInSection(kText, text_size_);
    data_start_ = session_->AllocateInSection(kData, data_size_);
    bss_start_ = session_->AllocateInSection(kBss, bss_size_);
    CHECK(text_start_ != nullptr && data_start_ != nullptr && bss_start_ != nullptr)
      << "Not enough space to load module on device";
    std::string relocated_bin = RelocateBinarySections(
        binary_,
        GetAddr(text_start_, lldevice_->base_addr()),
        GetAddr(data_start_, lldevice_->base_addr()),
        GetAddr(bss_start_, lldevice_->base_addr()));
    std::string text_contents = ReadSection(relocated_bin, kText);
    std::string data_contents = ReadSection(relocated_bin, kData);
    std::string bss_contents = ReadSection(relocated_bin, kBss);
    lldevice_->Write(text_start_, &text_contents[0], text_size_);
    lldevice_->Write(data_start_, &data_contents[0], data_size_);
    lldevice_->Write(bss_start_, &bss_contents[0], bss_size_);
    symbol_map_ = GetSymbolMap(relocated_bin);
  }
};

class MicroWrappedFunc {
 public:
  MicroWrappedFunc(MicroModuleNode* m,
                   const std::string& func_name,
                   void* func_addr) {
    m_ = m;
    func_name_ = func_name;
    func_addr_ = func_addr;
  }

  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    // no return value yet, but may implement in the future
    m_->RunFunction(func_name_, func_addr_, args);
  }

 private:
  // internal module
  MicroModuleNode* m_;
  // name of the function
  std::string func_name_;
  // address of the function to be called
  void* func_addr_;
};

PackedFunc MicroModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  void* func_addr = GetSymbol(symbol_map_, name, lldevice_->base_addr());
  MicroWrappedFunc f(this, name, func_addr);
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
