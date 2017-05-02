/*!
 *  Copyright (c) 2017 by Contributors
 *  Common build utilities
 * \file build_common.h
 */
#ifndef TVM_CODEGEN_BUILD_COMMON_H_
#define TVM_CODEGEN_BUILD_COMMON_H_

#include <tvm/codegen.h>
#include <unordered_map>
#include <string>
#include "../runtime/meta_data.h"

namespace tvm {
namespace codegen {
// Extract function information from device function.
inline std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const Array<LoweredFunc>& funcs) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;
  for (LoweredFunc f : funcs) {
    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->args.size(); ++i) {
      info.arg_types.push_back(Type2TVMType(f->args[i].type()));
    }
    for (size_t i = 0; i < f->thread_axis.size(); ++i) {
      info.thread_axis_tags.push_back(f->thread_axis[i]->thread_tag);
    }
    fmap[f->name] = info;
  }
  return fmap;
}
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_BUILD_COMMON_H_
