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
#include <unordered_map>
#include <utility>
#include <vector>

#include "../support/bytes_io.h"

namespace tvm {
namespace runtime {

ffi::json::Value FunctionInfo::SaveToJSON() const {
  namespace json = ::tvm::ffi::json;
  json::Object obj;
  obj.Set(ffi::String("name"), ffi::String(name));
  // arg_types: convert DLDataType to string
  json::Array sarg_types;
  for (const auto& t : arg_types) {
    sarg_types.push_back(ffi::String(DLDataTypeToString(t)));
  }
  obj.Set(ffi::String("arg_types"), std::move(sarg_types));
  {
    json::Array tags;
    for (const auto& s : launch_param_tags) tags.push_back(ffi::String(s));
    obj.Set(ffi::String("launch_param_tags"), std::move(tags));
  }
  // arg_extra_tags: convert enum to int
  json::Array iarg_extra_tags;
  for (const auto& t : arg_extra_tags) {
    iarg_extra_tags.push_back(static_cast<int64_t>(t));
  }
  obj.Set(ffi::String("arg_extra_tags"), std::move(iarg_extra_tags));
  return obj;
}

void FunctionInfo::LoadFromJSON(ffi::json::Object src) {
  namespace json = ::tvm::ffi::json;
  name = std::string(src.at(ffi::String("name")).cast<ffi::String>());
  // arg_types
  auto sarg_types_arr = src.at(ffi::String("arg_types")).cast<json::Array>();
  arg_types.resize(sarg_types_arr.size());
  for (size_t i = 0; i < sarg_types_arr.size(); ++i) {
    arg_types[i] = StringToDLDataType(std::string(sarg_types_arr[i].cast<ffi::String>()));
  }
  // launch_param_tags (optional, also support legacy "thread_axis_tags")
  auto lt = src.find(ffi::String("launch_param_tags"));
  if (lt != src.end()) {
    auto arr = (*lt).second.cast<json::Array>();
    launch_param_tags.clear();
    for (const auto& elem : arr) launch_param_tags.push_back(std::string(elem.cast<ffi::String>()));
  } else {
    auto tt = src.find(ffi::String("thread_axis_tags"));
    if (tt != src.end()) {
      auto arr = (*tt).second.cast<json::Array>();
      launch_param_tags.clear();
      for (const auto& elem : arr)
        launch_param_tags.push_back(std::string(elem.cast<ffi::String>()));
    }
  }
  // arg_extra_tags (optional)
  auto et = src.find(ffi::String("arg_extra_tags"));
  if (et != src.end()) {
    auto earr = (*et).second.cast<json::Array>();
    arg_extra_tags.resize(earr.size());
    for (size_t i = 0; i < earr.size(); ++i) {
      arg_extra_tags[i] = static_cast<ArgExtraTags>(earr[i].cast<int64_t>());
    }
  }
}

void FunctionInfo::Save(support::Stream* writer) const {
  writer->Write(name);
  writer->Write(arg_types);
  writer->Write(launch_param_tags);
  writer->Write(arg_extra_tags);
}

bool FunctionInfo::Load(support::Stream* reader) {
  if (!reader->Read(&name)) return false;
  if (!reader->Read(&arg_types)) return false;
  if (!reader->Read(&launch_param_tags)) return false;
  if (!reader->Read(&arg_extra_tags)) return false;
  return true;
}

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
  ICHECK(!fs.fail()) << "Cannot open " << file_name;
  // get its size:
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data->resize(size);
  fs.read(&(*data)[0], size);
}

void SaveBinaryToFile(const std::string& file_name, const std::string& data) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.write(&data[0], data.length());
}

void SaveMetaDataToFile(const std::string& file_name,
                        const std::unordered_map<std::string, FunctionInfo>& fmap) {
  namespace json = ::tvm::ffi::json;
  json::Object root;
  root.Set(ffi::String("tvm_version"), ffi::String("0.1.0"));
  json::Object func_info;
  for (const auto& kv : fmap) {
    func_info.Set(ffi::String(kv.first), kv.second.SaveToJSON());
  }
  root.Set(ffi::String("func_info"), std::move(func_info));
  std::ofstream fs(file_name.c_str());
  ICHECK(!fs.fail()) << "Cannot open file " << file_name;
  fs << std::string(json::Stringify(root, 2));
  fs.close();
}

void LoadMetaDataFromFile(const std::string& file_name,
                          std::unordered_map<std::string, FunctionInfo>* fmap) {
  namespace json = ::tvm::ffi::json;
  std::ifstream fs(file_name.c_str());
  ICHECK(!fs.fail()) << "Cannot open file " << file_name;
  std::string content((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
  fs.close();
  auto root = json::Parse(content).cast<json::Object>();
  // tvm_version is ignored
  auto func_info_obj = root.at(ffi::String("func_info")).cast<json::Object>();
  for (const auto& kv : func_info_obj) {
    FunctionInfo info;
    info.LoadFromJSON(kv.second.cast<json::Object>());
    (*fmap)[std::string(kv.first.cast<ffi::String>())] = info;
  }
}

void RemoveFile(const std::string& file_name) {
  // FIXME: This doesn't check the return code.
  std::remove(file_name.c_str());
}

void CopyFile(const std::string& src_file_name, const std::string& dest_file_name) {
  std::ifstream src(src_file_name, std::ios::binary);
  ICHECK(src) << "Unable to open source file '" << src_file_name << "'";

  std::ofstream dest(dest_file_name, std::ios::binary | std::ios::trunc);
  ICHECK(dest) << "Unable to destination source file '" << src_file_name << "'";

  dest << src.rdbuf();

  src.close();
  dest.close();

  ICHECK(dest) << "File-copy operation failed."
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
  ICHECK(strm->Read(&header)) << "Invalid parameters file format";
  ICHECK(header == kTVMTensorListMagic) << "Invalid parameters file format";
  ICHECK(strm->Read(&reserved)) << "Invalid parameters file format";

  std::vector<std::string> names;
  ICHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  ICHECK(size == names.size()) << "Invalid parameters file format";
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
      .def("runtime.LoadParamsFromFile", [](const ffi::String& path) {
        tvm::runtime::SimpleBinaryFileStream strm(path, "rb");
        return LoadParams(&strm);
      });
}

}  // namespace runtime
}  // namespace tvm
