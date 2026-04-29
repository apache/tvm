/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file file_utils.cc
 */
#include "file_utils.h"

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/support/io.h>

#include <fstream>
#include <utility>
#include <vector>

#include "../support/bytes_io.h"

namespace tvm {
namespace runtime {

std::string GetFileFormat(const std::string& file_name, const std::string& format) {
  std::string fmt = format;
  if (fmt.length() == 0) {
    size_t pos = file_name.find_last_of(".");
    if (pos != std::string::npos) {
      return file_name.substr(pos + 1, file_name.length() - pos - 1);
    } else {
      return "";
    }
  } else {
    return format;
  }
}

std::string GetCacheDir() {
  char* env_cache_dir;
  if ((env_cache_dir = getenv("TVM_CACHE_DIR"))) return env_cache_dir;
  if ((env_cache_dir = getenv("XDG_CACHE_HOME"))) {
    return std::string(env_cache_dir) + "/tvm";
  }
  if ((env_cache_dir = getenv("HOME"))) {
    return std::string(env_cache_dir) + "/.cache/tvm";
  }
  return ".";
}

std::string GetFileBasename(const std::string& file_name) {
  size_t last_slash = file_name.find_last_of("/");
  if (last_slash == std::string::npos) return file_name;
  return file_name.substr(last_slash + 1);
}

std::string GetMetaFilePath(const std::string& file_name) {
  size_t pos = file_name.find_last_of(".");
  if (pos != std::string::npos) {
    return file_name.substr(0, pos) + ".tvm_meta.json";
  } else {
    return file_name + ".tvm_meta.json";
  }
}

void LoadBinaryFromFile(const std::string& file_name, std::string* data) {
  std::ifstream fs(file_name, std::ios::in | std::ios::binary);
  TVM_FFI_ICHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

void SaveBinaryToFile(const std::string& file_name, const std::string& data) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  TVM_FFI_ICHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.write(&data[0], data.length());
}

void SaveMetaDataToFile(const std::string& file_name,
                        const ffi::Map<ffi::String, FunctionInfo>& fmap) {
  namespace json = ::tvm::ffi::json;
  json::Object root;
  root.Set("tvm_version", ffi::String("0.1.0"));
  json::Object func_info;
  for (const auto& kv : fmap) {
    func_info.Set(kv.first, kv.second->SaveToJSON());
  }
  root.Set("func_info", std::move(func_info));
  std::ofstream fs(file_name.c_str());
  TVM_FFI_ICHECK(!fs.fail()) << "Cannot open file " << file_name;
  fs << std::string(json::Stringify(root, 2));
  fs.close();
}

void LoadMetaDataFromFile(const std::string& file_name, ffi::Map<ffi::String, FunctionInfo>* fmap) {
  namespace json = ::tvm::ffi::json;
  std::ifstream fs(file_name.c_str());
  TVM_FFI_ICHECK(!fs.fail()) << "Cannot open file " << file_name;
  std::string content((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
  fs.close();
  auto root = json::Parse(content).cast<json::Object>();
  // tvm_version is ignored
  auto func_info_obj = root.at("func_info").cast<json::Object>();
  for (const auto& kv : func_info_obj) {
    auto info_node = ffi::make_object<FunctionInfoObj>();
    info_node->LoadFromJSON(kv.second.cast<json::Object>());
    fmap->Set(kv.first.cast<ffi::String>(), FunctionInfo(std::move(info_node)));
  }
}

void RemoveFile(const std::string& file_name) {
  // FIXME: This doesn't check the return code.
  std::remove(file_name.c_str());
}

void CopyFile(const std::string& src_file_name, const std::string& dest_file_name) {
  std::ifstream src(src_file_name, std::ios::binary);
  TVM_FFI_ICHECK(src) << "Unable to open source file '" << src_file_name << "'";

  std::ofstream dest(dest_file_name, std::ios::binary | std::ios::trunc);
  TVM_FFI_ICHECK(dest) << "Unable to destination source file '" << src_file_name << "'";

  dest << src.rdbuf();

  src.close();
  dest.close();

  TVM_FFI_ICHECK(dest) << "File-copy operation failed."
                       << " src='" << src_file_name << "'"
                       << " dest='" << dest_file_name << "'";
}

ffi::Map<ffi::String, Tensor> LoadParams(const std::string& param_blob) {
  support::BytesInStream strm(param_blob);
  return LoadParams(&strm);
}
ffi::Map<ffi::String, Tensor> LoadParams(support::Stream* strm) {
  ffi::Map<ffi::String, Tensor> params;
  uint64_t header, reserved;
  TVM_FFI_ICHECK(strm->Read(&header)) << "Invalid parameters file format";
  TVM_FFI_ICHECK(header == kTVMTensorListMagic) << "Invalid parameters file format";
  TVM_FFI_ICHECK(strm->Read(&reserved)) << "Invalid parameters file format";

  std::vector<std::string> names;
  TVM_FFI_ICHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  TVM_FFI_ICHECK(size == names.size()) << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    // The data_entry is allocated on device, Tensor.load always load the array into CPU.
    Tensor temp;
    temp.Load(strm);
    params.Set(names[i], temp);
  }
  return params;
}

void SaveParams(support::Stream* strm, const ffi::Map<ffi::String, Tensor>& params) {
  std::vector<std::string> names;
  std::vector<const DLTensor*> arrays;
  for (auto& p : params) {
    names.push_back(p.first);
    arrays.push_back(p.second.operator->());
  }

  uint64_t header = kTVMTensorListMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  strm->Write(names);
  {
    uint64_t sz = static_cast<uint64_t>(arrays.size());
    strm->Write(sz);
    for (size_t i = 0; i < sz; ++i) {
      tvm::runtime::SaveDLTensor(strm, arrays[i]);
    }
  }
}

std::string SaveParams(const ffi::Map<ffi::String, Tensor>& params) {
  std::string result;
  support::BytesOutStream strm(&result);
  SaveParams(&strm, params);
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<FunctionInfoObj>()
      .def_ro("name", &FunctionInfoObj::name)
      .def_ro("arg_types", &FunctionInfoObj::arg_types)
      .def_ro("launch_param_tags", &FunctionInfoObj::launch_param_tags)
      .def_ro("arg_extra_tags", &FunctionInfoObj::arg_extra_tags);
  refl::GlobalDef()
      .def("runtime.SaveParams",
           [](const ffi::Map<ffi::String, Tensor>& params) {
             std::string s = ::tvm::runtime::SaveParams(params);
             return ffi::Bytes(std::move(s));
           })
      .def("runtime.SaveParamsToFile",
           [](const ffi::Map<ffi::String, Tensor>& params, const ffi::String& path) {
             tvm::runtime::SimpleBinaryFileStream strm(path, "wb");
             SaveParams(&strm, params);
           })
      .def("runtime.LoadParams", [](const ffi::Bytes& s) { return ::tvm::runtime::LoadParams(s); })
      .def("runtime.LoadParamsFromFile",
           [](const ffi::String& path) {
             tvm::runtime::SimpleBinaryFileStream strm(path, "rb");
             return LoadParams(&strm);
           })
      // Registry: "runtime.LoadMetaDataFromJSON" — parse a tvm_meta.json
      // string into Map<String, FunctionInfo>.  Used by Python callers that
      // build FunctionInfo maps in-memory (e.g. tirx external_kernel) without
      // a disk round-trip.
      .def("runtime.LoadMetaDataFromJSON", [](const ffi::String& json_str) {
        namespace json = ::tvm::ffi::json;
        ffi::Map<ffi::String, FunctionInfo> fmap;
        auto root = json::Parse(std::string(json_str)).cast<json::Object>();
        auto func_info_obj = root.at("func_info").cast<json::Object>();
        for (const auto& kv : func_info_obj) {
          auto info_node = ffi::make_object<FunctionInfoObj>();
          info_node->LoadFromJSON(kv.second.cast<json::Object>());
          fmap.Set(kv.first.cast<ffi::String>(), FunctionInfo(std::move(info_node)));
        }
        return fmap;
      });
}

}  // namespace runtime
}  // namespace tvm
