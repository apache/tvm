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
 * Reflection utilities.
 * \file node/reflection.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/node.h>

namespace tvm {

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;

// API function to make node.
// args format:
//   key1, value1, ..., key_n, value_n
void MakeNode(const ffi::PackedArgs& args, ffi::Any* rv) {
  // TODO(tvm-team): consider further simplify by removing DictAttrsNode special handling
  ffi::String type_key = args[0].cast<ffi::String>();
  int32_t type_index;
  TVMFFIByteArray type_key_array = TVMFFIByteArray{type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  if (type_index == DictAttrsNode::RuntimeTypeIndex()) {
    ObjectPtr<DictAttrsNode> attrs = ffi::make_object<DictAttrsNode>();
    attrs->InitByPackedArgs(args.Slice(1), false);
    *rv = ObjectRef(attrs);
  } else {
    auto fcreate_object = ffi::Function::GetGlobalRequired("ffi.MakeObjectFromPackedArgs");
    fcreate_object.CallPacked(args, rv);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("node.MakeNode", MakeNode);
}

}  // namespace tvm
