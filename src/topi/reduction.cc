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
 * \brief Registration of reduction operators
 * \file reduction.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/utils.h>

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("topi.sum",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::sum(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                    args[2].cast<bool>());
                  })
      .def_packed("topi.min",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::min(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                    args[2].cast<bool>());
                  })
      .def_packed("topi.max",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::max(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                    args[2].cast<bool>());
                  })
      .def_packed("topi.argmin",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::argmin(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                       args[2].cast<bool>(), false, args[3].cast<bool>());
                  })
      .def_packed("topi.argmax",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::argmax(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                       args[2].cast<bool>(), false, args[3].cast<bool>());
                  })
      .def_packed("topi.prod",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::prod(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                     args[2].cast<bool>());
                  })
      .def_packed("topi.all",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::all(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                    args[2].cast<bool>());
                  })
      .def_packed("topi.any",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    *rv = topi::any(args[0].cast<te::Tensor>(), ArrayOrInt(args[1]),
                                    args[2].cast<bool>());
                  })
      .def_packed("topi.collapse_sum", [](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = topi::collapse_sum(args[0].cast<te::Tensor>(), args[1].cast<ffi::Array<Integer>>());
      });
}

}  // namespace topi
}  // namespace tvm
