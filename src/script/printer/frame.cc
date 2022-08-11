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
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/frame.h>

namespace tvm {
namespace script {
namespace printer {

MetadataFrame::MetadataFrame() : MetadataFrame(make_object<MetadataFrameNode>()) {}

VarDefFrame::VarDefFrame() : VarDefFrame(make_object<VarDefFrameNode>()) {}

TVM_REGISTER_NODE_TYPE(FrameNode);
TVM_REGISTER_GLOBAL("script.printer.FrameAddExitCallback")
    .set_body_typed([](Frame frame, runtime::TypedPackedFunc<void()> callback) {
      frame->AddExitCallback(callback);
    });
TVM_REGISTER_GLOBAL("script.printer.FrameEnterWithScope")
    .set_body_method<Frame>(&FrameNode::EnterWithScope);
TVM_REGISTER_GLOBAL("script.printer.FrameExitWithScope")
    .set_body_method<Frame>(&FrameNode::ExitWithScope);

TVM_REGISTER_NODE_TYPE(MetadataFrameNode);
TVM_REGISTER_GLOBAL("script.printer.MetadataFrame").set_body_typed([]() {
  return MetadataFrame();
});

TVM_REGISTER_NODE_TYPE(VarDefFrameNode);
TVM_REGISTER_GLOBAL("script.printer.VarDefFrame").set_body_typed([]() { return VarDefFrame(); });

}  // namespace printer
}  // namespace script
}  // namespace tvm
