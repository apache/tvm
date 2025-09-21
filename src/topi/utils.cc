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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/topi/detail/tensor_utils.h>

namespace tvm {
namespace topi {
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.utils.is_empty_shape",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::detail::is_empty_shape(args[0].cast<ffi::Array<PrimExpr>>());
                  })
      .def_packed("topi.utils.bilinear_sample_nchw",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = detail::bilinear_sample_nchw(
                        args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<PrimExpr>>(),
                        args[2].cast<PrimExpr>(), args[3].cast<PrimExpr>());
                  })
      .def_packed("topi.utils.bilinear_sample_nhwc", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = detail::bilinear_sample_nhwc(args[0].cast<te::Tensor>(),
                                           args[1].cast<ffi::Array<PrimExpr>>(),
                                           args[2].cast<PrimExpr>(), args[3].cast<PrimExpr>());
      });
}

}  // namespace topi
}  // namespace tvm
