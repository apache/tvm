/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen.cc
 * \brief Common utilities to generated C style code.
 */
#include <tvm/codegen.h>
#include <tvm/ir_pass.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>

namespace tvm {
namespace codegen {

runtime::Module Build(const Array<LoweredFunc>& funcs,
                      const std::string& target) {
  std::string mode = target;
  size_t pos = mode.find("-");
  if (pos != std::string::npos) {
    mode = mode.substr(0, pos);
  }
  std::string build_f_name = "codegen.build_" + mode;
  // Lower intrinsic functions
  Array<LoweredFunc> func_list;
  for (LoweredFunc f : funcs) {
    func_list.push_back(ir::LowerIntrin(f, target));
  }
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr)
      << "Target " << target << " is not enabled";
  runtime::Module m = (*bf)(func_list, target);
  return m;
}

}  // namespace codegen
}  // namespace tvm
