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
#include <tvm/relax/global_info.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<relax::DummyGlobalInfo>(
      "", [](GlobalInfo ginfo, AccessPath p, IRDocsifier d) -> Doc {
        return Relax(d, "dummy_global_info")->Call({});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<relax::VDevice>(
      "", [](relax::VDevice vdev, AccessPath p, IRDocsifier d) -> Doc {
        d->AddGlobalInfo("vdevice", vdev);
        ffi::Map<ffi::String, ffi::Any> config = vdev->target->ToConfig();
        return Relax(d, "vdevice")
            ->Call({d->AsDoc<ExprDoc>(config, p),
                    LiteralDoc::Int(vdev->vdevice_id, p->Attr("vdevice_id")),
                    LiteralDoc::Str(vdev->memory_scope, p->Attr("memory_scope"))});
      });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
