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
#ifndef TVM_RUNTIME_MODULE_H_
#define TVM_RUNTIME_MODULE_H_

#include <string.h>
#include <tvm/runtime/c_runtime_api.h>

struct packed_func_t;
typedef struct packed_func_t PackedFunc;

/*!
 * \brief Module container of TVM.
 */
typedef struct module_t {
  /*!
   * \brief Get packed function from current module by name.
   *
   * \param name The name of the function.
   * \param query_imports Whether also query dependency modules.
   * \return The result function.
   *  This function will return PackedFunc(nullptr) if function do not exist.
   * \note Implemented in packed_func.cc
   */
  void (*GetFunction)(const char * name, PackedFunc * pf);
  void (*set_input)(const struct module_t * mod, const char * name, DLTensor * data);
  void (*load_params)(const struct module_t * mod, const TVMByteArray * params_arr);
  void (*run)(const struct module_t * mod);
} Module;

#endif  // TVM_RUNTIME_MODULE_H_
