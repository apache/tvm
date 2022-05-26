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
 * \file src/runtime/contrib/dnnl/dnnl_utils.cc
 */

#include "dnnl_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {
using dt = dnnl::memory::data_type;
dt dtype_dl2dnnl(DLDataType dltype) {
  dt dnnl_type = dt::undef;
  if (dltype.code == DataType::TypeCode::kFloat) {
    if (dltype.bits == 16) {
      dnnl_type = dt::f16;
    } else if (dltype.bits == 32) {
      dnnl_type = dt::f32;
    }
  } else if (dltype.code == DataType::TypeCode::kBFloat && dltype.bits == 16) {
    dnnl_type = dt::bf16;
  } else if (dltype.code == DataType::TypeCode::kInt) {
    if (dltype.bits == 8) {
      dnnl_type = dt::s8;
    } else if (dltype.bits == 32) {
      dnnl_type = dt::s32;
    }
  } else if (dltype.code == DataType::TypeCode::kUInt && dltype.bits == 8) {
    dnnl_type = dt::u8;
  }
  if (dnnl_type == dt::undef) {
    LOG_ERROR << "unsupported datatype: code=" << dltype.code << ", bits=" << dltype.bits;
  }
  return dnnl_type;
}
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
