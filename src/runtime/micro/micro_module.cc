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

namespace tvm {
namespace runtime {
/*!
 * \brief module for uTVM micro devices
 */
class MicroModuleNode final : public ModuleNode {
 public:
  ~MicroModuleNode();

  const char* type_key() const final {
    return "micro";
  }

  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  /*!
   * \brief initializes module by establishing device connection and loads binary
   * \param binary name of the binary to be loaded
   */
  void InitMicroModule(const std::string binary);

  /*!
   * \brief runs selected function on the micro device
   * \param func name of the function to be run
   * \param args type-erased arguments passed to the function
   */
  void RunFunction(std::string func, TVMArgs args);

 private:
  /*! \brief loaded module text start address */
  void* text_start_;
  /*! \brief loaded module data start address */
  void* data_start_;
  /*! \brief loaded module bss start address */
  void* bss_start_;
  /*! \brief size of module text section */
  size_t code_size_;
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
  std::unordered_map<std::string, void*> symbol_map;
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

  void operator()(TVMArgs args, TVMRetValue* rv) const {
  }

 private:
  // internal module
  MicroModuleNode* m_;
  // name of the function
  std::string func_name_;
  // address of the function to be called
  void* func_addr_;
};

// TODO: register module load function
// register loadfile function to load module from Python frontend
TVM_REGISTER_GLOBAL("module.loadfile_micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  });
}  // namespace runtime
}  // namespace tvm
