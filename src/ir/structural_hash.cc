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
 * \file src/ir/structural_hash.cc
 */
#include <tvm/ffi/extra/base64.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/io.h>
#include <tvm/target/codegen.h>

#include <algorithm>
#include <unordered_map>

#include "../support/base64.h"
#include "../support/bytes_io.h"
#include "../support/str_escape.h"
#include "../support/utils.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("node.StructuralHash",
                        [](const Any& object, bool map_free_vars) -> int64_t {
                          return ffi::StructuralHash::Hash(object, map_free_vars);
                        });
  refl::TypeAttrDef<ffi::ModuleObj>()
      .def("__data_to_json__",
           [](const ffi::ModuleObj* node) {
             std::string bytes = codegen::SerializeModuleToBytes(ffi::GetRef<ffi::Module>(node),
                                                                 /*export_dso*/ false);
             return ffi::Base64Encode(ffi::Bytes(bytes));
           })
      .def("__data_from_json__", [](const ffi::String& base64_bytes) {
        ffi::Bytes bytes = ffi::Base64Decode(base64_bytes);
        ffi::Module rtmod = codegen::DeserializeModuleFromBytes(bytes.operator std::string());
        return rtmod;
      });

  refl::TypeAttrDef<runtime::Tensor::Container>()
      .def("__data_to_json__",
           [](const runtime::Tensor::Container* node) {
             std::string result;
             support::BytesOutStream mstrm(&result);
             support::Base64OutStream b64strm(&mstrm);
             runtime::SaveDLTensor(&b64strm, node);
             b64strm.Finish();
             return ffi::String(std::move(result));
           })
      .def("__data_from_json__", [](const std::string& blob) {
        support::BytesInStream mstrm(blob);
        support::Base64InStream b64strm(&mstrm);
        b64strm.InitPosition();
        runtime::Tensor temp;
        TVM_FFI_ICHECK(temp.Load(&b64strm));
        return temp;
      });
}

struct RefToObjectPtr : public ObjectRef {
  static ObjectPtr<Object> Get(const ObjectRef& ref) {
    return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(ref);
  }
};

}  // namespace tvm
