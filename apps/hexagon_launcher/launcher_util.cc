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

#include "launcher_util.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <utility>

size_t get_file_size(std::ifstream& in_file) {
  std::ifstream::pos_type pos = in_file.tellg();
  size_t size = in_file.seekg(0, std::ios::end).tellg();
  in_file.seekg(pos, std::ios::beg);
  return size;
}

size_t get_file_size(std::ifstream&& in_file) {
  return get_file_size(in_file);  // calls the & version
}

std::string load_text_file(const std::string& file_name) {
  constexpr size_t block_size = 1024 * 1024;  // 1MB
  std::ifstream in_file(file_name);
  ICHECK(in_file.is_open()) << "cannot open file " << file_name;
  size_t file_size = get_file_size(in_file);
  std::string buffer(file_size + 1, 0);

  in_file.read(&buffer[0], file_size);
  return std::move(buffer);
}

void* load_binary_file(const std::string& file_name, void* buffer, size_t buffer_size) {
  std::ifstream in_file(file_name);
  ICHECK(in_file.is_open()) << "cannot open file " << file_name;
  size_t file_size = get_file_size(in_file);

  in_file.read(reinterpret_cast<std::ifstream::char_type*>(buffer),
               std::min(buffer_size, file_size));
  return buffer;
}

void write_binary_file(const std::string& file_name, void* buffer, size_t buffer_size) {
  std::ofstream out_file(file_name);
  ICHECK(out_file.is_open()) << "cannot open file " << file_name;

  out_file.write(reinterpret_cast<std::ofstream::char_type*>(buffer), buffer_size);
}
