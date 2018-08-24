/*!
 *  Copyright (c) 2018 by Contributors
 * \file aocl_module.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "aocl_common.h"
#include "aocl_module.h"

namespace tvm {
namespace runtime {

class AOCLModuleNode : public OpenCLModuleNode {
 public:
  explicit AOCLModuleNode(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string source)
      : OpenCLModuleNode(data, fmt, fmap, source) {}
  const std::shared_ptr<cl::OpenCLWorkspace>& GetGlobalWorkspace() final;
};

const std::shared_ptr<cl::OpenCLWorkspace>& AOCLModuleNode::GetGlobalWorkspace() {
  return cl::AOCLWorkspace::Global();
}

Module AOCLModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source) {
  std::shared_ptr<AOCLModuleNode> n =
      std::make_shared<AOCLModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}

Module AOCLModuleLoadFile(const std::string& file_name,
                          const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return AOCLModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("module.loadfile_aocx")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = AOCLModuleLoadFile(args[0], args[1]);
  });

}  // namespace runtime
}  // namespace tvm
