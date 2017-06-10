/*!
 *  Copyright (c) 2017 by Contributors
 * \file source_module.cc
 * \brief Source code module, only for viewing
 */
#include <tvm/runtime/packed_func.h>
#include "./codegen_source_base.h"

namespace tvm {
namespace codegen {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;
// Simulator function
class SourceModuleNode final : public runtime::ModuleNode {
 public:
  SourceModuleNode(std::string code,
                   std::string fmt)
      : code_(code), fmt_(fmt) {}
  const char* type_key() const {
    return "source";
  }
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
    return code_;
  }

 private:
  std::string code_;
  std::string fmt_;
};

runtime::Module SourceModuleCreate(std::string code, std::string fmt) {
  std::shared_ptr<SourceModuleNode> n =
      std::make_shared<SourceModuleNode>(code, fmt);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("module.source_module_create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = SourceModuleCreate(args[0], args[1]);
  });
}  // namespace codegen
}  // namespace tvm
