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
 * \file tvm/c_dsl_api.h
 *
 * \brief TVM DSL Node C API, used to interact to DSL compilation.
 *
 *  These are only a few functions needed for DSL construction time.
 *  These function are only available when link libtvm.
 *  If only TVM runtime is linked, calling these function will trigger error.
 *
 * \note Most API functions are registerd as PackedFunc and
 *  can be grabbed via TVMFuncGetGlobal
 */
#ifndef TVM_C_DSL_API_H_
#define TVM_C_DSL_API_H_

#include "runtime/c_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief handle to node */
typedef void* NodeHandle;

/*!
 * \brief free the node handle
 * \param handle The node handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMNodeFree(NodeHandle handle);

/*!
 * \brief Convert type key to type index.
 * \param type_key The key of the type.
 * \param out_index the corresponding type index.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMNodeTypeKey2Index(const char* type_key,
                                 int* out_index);

/*!
 * \brief Get runtime type index of the node.
 * \param handle the node handle.
 * \param out_index the corresponding type index.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMNodeGetTypeIndex(NodeHandle handle,
                                int* out_index);

/*!
 * \brief get attributes given key
 * \param handle The node handle
 * \param key The attribute name
 * \param out_value The attribute value
 * \param out_type_code The type code of the attribute.
 * \param out_success Whether get is successful.
 * \return 0 when success, -1 when failure happens
 * \note API calls always exchanges with type bits=64, lanes=1
 */
TVM_DLL int TVMNodeGetAttr(NodeHandle handle,
                           const char* key,
                           TVMValue* out_value,
                           int* out_type_code,
                           int* out_success);

/*!
 * \brief get attributes names in the node.
 * \param handle The node handle
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
TVM_DLL int TVMNodeListAttrNames(NodeHandle handle,
                                 int *out_size,
                                 const char*** out_array);
#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_C_DSL_API_H_
