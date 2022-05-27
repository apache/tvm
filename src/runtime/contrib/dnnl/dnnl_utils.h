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
 * \file src/runtime/contrib/dnnl/dnnl_utils.h
 * \brief utils for DNNL.
 */

#ifndef TVM_RUNTIME_CONTRIB_DNNL_DNNL_UTILS_H_
#define TVM_RUNTIME_CONTRIB_DNNL_DNNL_UTILS_H_

#include <tvm/runtime/data_type.h>

#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief Convert a DLPack data type to a DNNL data type.
 * \param dltype The DLPack data type.
 * \return The corresponding DNNL data type.
 */
dnnl::memory::data_type dtype_dl2dnnl(DLDataType dltype);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_DNNL_DNNL_UTILS_H_
