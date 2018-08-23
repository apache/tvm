/*!
 *  Copyright (c) 2017 by Contributors
 * \file source_module.cc
 * \brief Source code module, only for viewing
 */
#include <tvm/runtime/packed_func.h>
#include "codegen_source_base.h"
#include "../runtime/file_util.h"
#include "../runtime/meta_data.h"

namespace tvm {
namespace codegen {

using runtime::TVMArgs;
using runtime::TVMRetValue;
using runtime::PackedFunc;

using runtime::GetFileFormat;
using runtime::GetMetaFilePath;
using runtime::FunctionInfo;
using runtime::SaveBinaryToFile;

// Simulator function
class SourceModuleNode : public runtime::ModuleNode {
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

 protected:
  std::string code_;
  std::string fmt_;
};

runtime::Module SourceModuleCreate(std::string code, std::string fmt) {
  std::shared_ptr<SourceModuleNode> n =
      std::make_shared<SourceModuleNode>(code, fmt);
  return runtime::Module(n);
}

// supports limited save without cross compile
class DeviceSourceModuleNode final : public runtime::ModuleNode {
 public:
  DeviceSourceModuleNode(std::string data,
                         std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap,
                         std::string type_key,
                         std::function<std::string(const std::string&)> fget_source)
    : data_(data),
      fmt_(fmt),
      fmap_(fmap),
      type_key_(type_key),
      fget_source_(fget_source) {}

  PackedFunc GetFunction(
        const std::string& name,
        const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    LOG(FATAL) << "Source module cannot execute, to get executable module"
               << " build TVM with \'" << fmt_ << "\' runtime support";
    return PackedFunc();
  }

  std::string GetSource(const std::string& format) final {
    if (fget_source_ != nullptr) {
      return fget_source_(format);
    } else {
      return data_;
    }
  }

  const char* type_key() const {
    return type_key_.c_str();
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, fmt_)
        << "Can only save to format=" << fmt_;
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

 private:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::string type_key_;
  std::function<std::string(const std::string&)> fget_source_;
};

runtime::Module DeviceSourceModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string type_key,
    std::function<std::string(const std::string&)> fget_source) {
  std::shared_ptr<DeviceSourceModuleNode> n =
      std::make_shared<DeviceSourceModuleNode>(data, fmt, fmap, type_key, fget_source);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("module.source_module_create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = SourceModuleCreate(args[0], args[1]);
  });
}  // namespace codegen
}  // namespace tvm
