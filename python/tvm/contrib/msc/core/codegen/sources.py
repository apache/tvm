# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""tvm.contrib.msc.core.codegen.sources"""

from typing import Dict


def get_base_h_code() -> str:
    """Create base header file codes

    Returns
    -------
    source: str
        The base header source.
    """

    return """#ifndef TVM_CONTRIB_MSC_UTILS_BASE_H_
#define TVM_CONTRIB_MSC_UTILS_BASE_H_

#include <cassert>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace contrib {
namespace msc {

class CommonUtils {
 public:
  template <typename T>
  static bool CompareBuffers(const T* golden, const T* result, size_t size) {
    return true;
  }
};

class FileUtils {
 public:
  static bool FileExist(const std::string& file);

  template <typename T>
  static bool ReadToBuffer(const std::string& file, T* buffer, size_t size) {
    std::ifstream in_file(file, std::ifstream::binary);
    if (!in_file.is_open()) {
      return false;
    }
    try {
      in_file.read((char*)(&buffer[0]), size * sizeof(T));
    } catch (std::exception const& e) {
      in_file.close();
      return false;
    }
    in_file.close();
    return true;
  }
};

class DatasetReader {
 public:
  DatasetReader(const std::string& folder, int max_size = -1);

  void Reset();

  bool ReadNext(void* buffers[], int num_datas = -1);

  const std::vector<std::string> GetTensorNames() { return tensor_names_; }

  size_t GetTensorSize(const std::string& name);

  const std::string GetSaveName(const std::string& name);

 private:
  std::string folder_;
  size_t max_size_;
  size_t cur_cnt_;
  std::vector<std::string> tensor_names_;
  std::unordered_map<std::string, std::string> save_names_;
  std::unordered_map<std::string, size_t> tensor_sizes_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_UTILS_BASE_H_
"""


def get_base_cc_code() -> str:
    """Create base cc file codes

    Returns
    -------
    source: str
        The base cc source.
    """

    return """#include "base.h"

#include <algorithm>
#include <fstream>

namespace tvm {
namespace contrib {
namespace msc {

bool FileUtils::FileExist(const std::string& file) {
  std::ifstream in_file(file, std::ifstream::binary);
  if (in_file.is_open()) {
    in_file.close();
    return true;
  }
  return false;
}

DatasetReader::DatasetReader(const std::string& folder, int max_size) {
  folder_ = folder;
  const std::string info_file = folder_ + "/datas_info.txt";
  std::ifstream input(info_file, std::ios::binary);
  assert(input.is_open() && ("Failed to open file " + info_file).c_str());
  std::string line;
  while (getline(input, line)) {
    // define name
    int pos = line.find(" ");
    assert(pos > 0 && ("Can not find space in line " + line).c_str());
    const auto& name = line.substr(0, pos);
    tensor_names_.push_back(name);
    const auto& left = line.substr(pos + 1, line.size());
    // define save_name
    pos = left.find(" ");
    assert(pos > 0 && ("Can not find space in left " + left).c_str());
    save_names_[name] = left.substr(0, pos);
    // define size
    const auto& byte_size = left.substr(pos + 1, left.size());
    tensor_sizes_[name] = static_cast<size_t>(std::stoi(byte_size));
  }
  size_t file_cnt = 0;
  while (true) {
    bool all_exists = true;
    for (const auto& pair : save_names_) {
      const auto& d_file =
          folder_ + "/" + pair.second + "/batch_" + std::to_string(file_cnt) + ".bin";
      if (!FileUtils::FileExist(d_file)) {
        all_exists = false;
        break;
      }
    }
    if (!all_exists) {
      break;
    }
    file_cnt++;
  }
  max_size_ = max_size > 0 ? static_cast<size_t>(max_size) : file_cnt;
  max_size_ = std::min(max_size_, file_cnt);
  Reset();
}

void DatasetReader::Reset() { cur_cnt_ = 0; }

bool DatasetReader::ReadNext(void* buffers[], int num_datas) {
  if (cur_cnt_ >= max_size_) {
    return false;
  }
  size_t max_num = num_datas > 0 ? static_cast<size_t>(num_datas) : tensor_names_.size();
  max_num = std::min(max_num, tensor_names_.size());
  for (size_t i = 0; i < max_num; i++) {
    const auto& name = tensor_names_[i];
    const auto& d_file =
        folder_ + "/" + GetSaveName(name) + "/batch_" + std::to_string(cur_cnt_) + ".bin";
    if (!FileUtils::ReadToBuffer(d_file, (char*)buffers[i], GetTensorSize(name))) {
      return false;
    }
  }
  cur_cnt_++;
  return true;
}

size_t DatasetReader::GetTensorSize(const std::string& name) {
  assert(tensor_sizes_.count(name));
  return tensor_sizes_[name];
}

const std::string DatasetReader::GetSaveName(const std::string& name) {
  assert(save_names_.count(name));
  return save_names_[name];
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
"""


def get_base_sources() -> Dict[str, str]:
    """Create base sources for cpp codegen

    Returns
    -------
    sources: dict<str,str>
        The base utils sources.
    """

    return {"base.h": get_base_h_code(), "base.cc": get_base_cc_code()}
