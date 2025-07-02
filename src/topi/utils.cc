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
 * \brief Registration of utils operators
 * \file utils.cc
 */

#include <tvm/ffi/function.h>
#include <tvm/topi/detail/tensor_utils.h>

namespace tvm {
namespace topi {
TVM_FFI_REGISTER_GLOBAL("topi.utils.is_empty_shape")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = topi::detail::is_empty_shape(args[0].cast<Array<PrimExpr>>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.utils.bilinear_sample_nchw")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv =
          detail::bilinear_sample_nchw(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                                       args[2].cast<PrimExpr>(), args[3].cast<PrimExpr>());
    });

TVM_FFI_REGISTER_GLOBAL("topi.utils.bilinear_sample_nhwc")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv =
          detail::bilinear_sample_nhwc(args[0].cast<te::Tensor>(), args[1].cast<Array<PrimExpr>>(),
                                       args[2].cast<PrimExpr>(), args[3].cast<PrimExpr>());
    });

}  // namespace topi
}  // namespace tvm
