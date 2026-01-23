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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>

namespace tvm {
namespace runtime {

struct SessionObj::FFI {
  static DRef CallWithPacked(Session sess, const ffi::PackedArgs& args) {
    return sess->CallWithPacked(args);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<DRefObj>();
  refl::ObjectDef<SessionObj>();
  refl::GlobalDef()
      .def("runtime.disco.SessionThreaded", Session::ThreadedSession)
      .def_method("runtime.disco.DRefDebugGetFromRemote", &DRefObj::DebugGetFromRemote)
      .def_method("runtime.disco.DRefDebugCopyFrom", &DRefObj::DebugCopyFrom)
      .def_method("runtime.disco.SessionGetNumWorkers", &SessionObj::GetNumWorkers)
      .def_method("runtime.disco.SessionGetGlobalFunc", &SessionObj::GetGlobalFunc)
      .def_method("runtime.disco.SessionCopyFromWorker0", &SessionObj::CopyFromWorker0)
      .def_method("runtime.disco.SessionCopyToWorker0", &SessionObj::CopyToWorker0)
      .def_method("runtime.disco.SessionSyncWorker", &SessionObj::SyncWorker)
      .def_method("runtime.disco.SessionInitCCL", &SessionObj::InitCCL)
      .def_packed("runtime.disco.SessionCallPacked",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    Session self = args[0].cast<Session>();
                    *rv = SessionObj::FFI::CallWithPacked(self, args.Slice(1));
                  })
      .def_method("runtime.disco.SessionShutdown", &SessionObj::Shutdown);
}

}  // namespace runtime
}  // namespace tvm
