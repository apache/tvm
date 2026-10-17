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

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/tirx/layout.h>

#include "../../../tirx/ir/layout/axis_registry.h"
namespace tvm::backend::trn {
void RegisterLayoutAxes() {
  for (const char* name : {"P", "F", "Bank"}) {
    tvm::tirx::AxisDef(name).set_attr("backend", ffi::String("trn")).set_attr("thread", false);
  }
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tirx.trn.RegisterLayoutAxes", RegisterLayoutAxes);
}
}  // namespace tvm::backend::trn
