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

#ifndef VTA_DPI_MODULE_H_
#define VTA_DPI_MODULE_H_

#include <tvm/runtime/module.h>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <string>

namespace vta {
namespace dpi {

class DPIModuleNode : public tvm::runtime::ModuleNode {
 public:
  virtual void Launch(uint64_t max_cycles) = 0;
  virtual void WriteReg(int addr, uint32_t value) = 0;
  virtual uint32_t ReadReg(int addr) = 0;
  virtual void Finish(uint32_t length) = 0;

  static tvm::runtime::Module Load(std::string dll_name);
};

}  // namespace dpi
}  // namespace vta
#endif  // VTA_DPI_MODULE_H_

