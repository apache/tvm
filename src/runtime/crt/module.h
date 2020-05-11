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
 * \file src/runtime/crt/module.h
 * \brief Runtime container of the functions
 */
#ifndef TVM_RUNTIME_CRT_MODULE_H_
#define TVM_RUNTIME_CRT_MODULE_H_

#include <string.h>
#include <tvm/runtime/c_runtime_api.h>

struct TVMPackedFunc;

/*!
 * \brief Module container of TVM.
 */
typedef struct TVMModule {
  /*!
   * \brief Get packed function from current module by name.
   *
   * \param name The name of the function.
   * \param pf The result function.
   *
   *  This function will return PackedFunc(nullptr) if function do not exist.
   */
  void (*GetFunction)(struct TVMModule* mod, const char* name, struct TVMPackedFunc* pf);
} TVMModule;

#endif  // TVM_RUNTIME_CRT_MODULE_H_
