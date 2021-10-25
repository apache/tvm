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

#ifndef TVM_RUNTIME_HEXAGON_LAUNCHER_LAUNCHER_UTIL_H_
#define TVM_RUNTIME_HEXAGON_LAUNCHER_LAUNCHER_UTIL_H_

#include <cstddef>
#include <fstream>
#include <string>

size_t get_file_size(std::ifstream& in_file);
size_t get_file_size(std::ifstream&& in_file);

std::string load_text_file(const std::string& file_name);
void* load_binary_file(const std::string& file_name, void* buffer, size_t buffer_size);
void write_binary_file(const std::string& file_name, void* buffer, size_t buffer_size);

#endif  // TVM_RUNTIME_HEXAGON_LAUNCHER_LAUNCHER_UTIL_H_
