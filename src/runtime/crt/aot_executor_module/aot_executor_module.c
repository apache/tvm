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

// LINT_C_FILE

/*!
 * \file aot_executor_module.c
 * \brief wrap aot_executor into a TVMModule for use with RPC.
 */

#include <tvm/runtime/crt/func_registry.h>
#include <tvm/runtime/crt/aot_executor.h>
#include <tvm/runtime/crt/aot_executor_module.h>
#include <tvm/runtime/crt/module.h>

int32_t TVMAotExecutorModule_Create(TVMValue* args, int* tcodes, int nargs, TVMValue* ret_values,
                                      int* ret_tcodes, void* resource_handle) {

  return kTvmErrorNoError;
}

tvm_crt_error_t TVMAotExecutorModule_Register() {

  return TVMFuncRegisterGlobal("tvm.aot_executor.create", &TVMAotExecutorModule_Create, 0);
}
