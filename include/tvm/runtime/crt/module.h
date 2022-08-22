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
 * \file include/tvm/runtime/crt/module.h
 * \brief Runtime container of the functions
 */
#ifndef TVM_RUNTIME_CRT_MODULE_H_
#define TVM_RUNTIME_CRT_MODULE_H_

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/crt/func_registry.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Module container of TVM.
 */
typedef struct TVMModule {
  /*! \brief The function registry associated with this module. */
  const TVMFuncRegistry* registry;
} TVMModule;

/*!
 * \brief Create a new module handle from the given TVMModule instance.
 * \param mod The module instance to register.
 * \param out_handle Pointer to receive the newly-minted handle for this module.
 * \return 0 on success, non-zero on error.
 */
int TVMModCreateFromCModule(const TVMModule* mod, TVMModuleHandle* out_handle);

/*! \brief Entry point for the system lib module. */
const TVMModule* TVMSystemLibEntryPoint(void);

#ifdef __cplusplus
}
#endif
#endif  // TVM_RUNTIME_CRT_MODULE_H_
