/*!
 *  Copyright (c) 2017 by Contributors
 * \file stack_vm_module.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <tvm/codegen.h>
#include "./codegen_stack_vm.h"

namespace tvm {
namespace codegen {

class StackVMModuleNode : public runtime::ModuleNode {
 public:
  const char* type_key() const {
    return "stackvm";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    if (name == runtime::symbol::tvm_module_main) {
      return GetFunction(entry_func_, sptr_to_self);
    }
    auto it = fmap_.find(name);
    if (it == fmap_.end()) return PackedFunc();
    const StackVM& vm = it->second;
    // capture sptr_to_self to keep module node alive.
    return PackedFunc([vm, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        vm(args);
      });
  }

  std::string GetSource(const std::string& format) final {
    std::ostringstream os;
    for (const auto& kv : fmap_) {
      os << "Function: " << kv.first << '\n';
      os << kv.second;
    }
    return os.str();
  }

  static runtime::Module Build(const Array<LoweredFunc>& funcs) {
    CHECK_NE(funcs.size(), 0U);
    std::shared_ptr<StackVMModuleNode> n =
        std::make_shared<StackVMModuleNode>();
    for (LoweredFunc f : funcs) {
      StackVM vm = codegen::CodeGenStackVM().Compile(f);
      CHECK(!n->fmap_.count(f->name))
          << "Function name " << f->name << "already exist in list";
      vm.mod_ctx = n.get();
      n->fmap_[f->name] = std::move(vm);
    }
    n->entry_func_ = funcs[0]->name;
    return runtime::Module(n);
  }

 private:
  // entry function.
  std::string entry_func_;
  // internal function map
  std::unordered_map<std::string, StackVM> fmap_;
};

TVM_REGISTER_API("codegen.build_stackvm")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = StackVMModuleNode::Build(args[0]);
  });

}  // namespace codegen
}  // namespace tvm
