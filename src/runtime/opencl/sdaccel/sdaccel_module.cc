/*!
 *  Copyright (c) 2018 by Contributors
 * \file sdaccel_module.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <vector>
#include <string>
#include <unordered_map>
#include "sdaccel_common.h"
#include "sdaccel_module.h"

namespace tvm {
namespace runtime {

class SDAccelModuleNode : public OpenCLModuleNode {
 public:
  explicit SDAccelModuleNode(std::string data,
                             std::string fmt,
                             std::unordered_map<std::string, FunctionInfo> fmap,
                             std::string source)
      : OpenCLModuleNode(data, fmt, fmap, source) {}
  const std::shared_ptr<cl::OpenCLWorkspace>& GetGlobalWorkspace() final;
};

const std::shared_ptr<cl::OpenCLWorkspace>& SDAccelModuleNode::GetGlobalWorkspace() {
  return cl::SDAccelWorkspace::Global();
}

Module SDAccelModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source) {
  std::shared_ptr<SDAccelModuleNode> n =
      std::make_shared<SDAccelModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}

Module SDAccelModuleLoadFile(const std::string& file_name,
                             const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return SDAccelModuleCreate(data, fmt, fmap, std::string());
}

Module SDAccelModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return SDAccelModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("module.loadfile_xclbin")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = SDAccelModuleLoadFile(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("module.loadfile_awsxclbin")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = SDAccelModuleLoadFile(args[0], args[1]);
  });
}  // namespace runtime
}  // namespace tvm
