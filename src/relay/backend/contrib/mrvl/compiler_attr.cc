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
 * \file src/relay/backend/contrib/mrvl/compiler_attr.cc
 * \brief Marvell MLIP specific attributes
 */

#include <stdlib.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace mrvl {

/*! \brief Attributes to store the compiler options for Mrvl MLIP */
struct MrvlCompilerConfigNode : public tvm::AttrsNode<MrvlCompilerConfigNode> {
  String mcpu;
  IntImm num_tiles;
  String mattr;

  TVM_DECLARE_ATTRS(MrvlCompilerConfigNode, "ext.attrs.MrvlCompilerConfigNode") {
    TVM_ATTR_FIELD(mcpu)
        .describe(
            "The CPU class of Marvell(R) ML Inference Processor;"
            "possible values = {cn10ka, cnf10kb}")
        .set_default("cn10ka");
    TVM_ATTR_FIELD(num_tiles)
        .describe("Maximum number of tiles that may be used, possible values = {1,2,4,8}")
        .set_default(IntImm(DataType::Int(64), 8));
    TVM_ATTR_FIELD(mattr)
        .describe("Attributes for MLIP; possible values = {quantize,wb_pin_ocm}")
        .set_default("");
  }
};

class MrvlCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(MrvlCompilerConfig, Attrs, MrvlCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(MrvlCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.mrvl.options", MrvlCompilerConfig);

TVM_REGISTER_TARGET_KIND("mrvl", kDLCPU);

}  // namespace mrvl
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
