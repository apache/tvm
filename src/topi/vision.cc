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
 * \brief Registration of vision operators
 * \file vision.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/topi/vision/reorg.h>

namespace tvm {
namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_FFI_REGISTER_GLOBAL("topi.vision.reorg")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = vision::reorg(args[0].cast<te::Tensor>(), args[1].cast<int>());
    });

}  // namespace topi
}  // namespace tvm
