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
#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

struct SessionObj::FFI {
  static DRef CallWithPacked(Session sess, const TVMArgs& args) {
    return sess->CallWithPacked(args);
  }
};

TVM_REGISTER_OBJECT_TYPE(DRefObj);
TVM_REGISTER_OBJECT_TYPE(SessionObj);
TVM_REGISTER_GLOBAL("runtime.disco.SessionThreaded").set_body_typed(Session::ThreadedSession);
TVM_REGISTER_GLOBAL("runtime.disco.DRefDebugGetFromRemote")
    .set_body_method<DRef>(&DRefObj::DebugGetFromRemote);
TVM_REGISTER_GLOBAL("runtime.disco.DRefDebugCopyFrom")
    .set_body_method<DRef>(&DRefObj::DebugCopyFrom);
TVM_REGISTER_GLOBAL("runtime.disco.SessionGetGlobalFunc")
    .set_body_method<Session>(&SessionObj::GetGlobalFunc);
TVM_REGISTER_GLOBAL("runtime.disco.SessionCopyFromWorker0")
    .set_body_method<Session>(&SessionObj::CopyFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.SessionCopyToWorker0")
    .set_body_method<Session>(&SessionObj::CopyToWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.SessionSyncWorker")
    .set_body_method<Session>(&SessionObj::SyncWorker);
TVM_REGISTER_GLOBAL("runtime.disco.SessionInitCCL")  //
    .set_body_method<Session>(&SessionObj::InitCCL);
TVM_REGISTER_GLOBAL("runtime.disco.SessionCallPacked").set_body([](TVMArgs args, TVMRetValue* rv) {
  Session self = args[0];
  *rv = SessionObj::FFI::CallWithPacked(
      self, TVMArgs(args.values + 1, args.type_codes + 1, args.num_args - 1));
});

}  // namespace runtime
}  // namespace tvm
