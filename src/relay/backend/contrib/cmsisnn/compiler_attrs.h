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
 * \file src/relay/backend/contrib/cmsisnn/compiler_attrs.h
 * \brief CMSIS-NN Compiler Attribute functionality
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_CMSISNN_COMPILER_ATTRS_H_
#define TVM_RELAY_BACKEND_CONTRIB_CMSISNN_COMPILER_ATTRS_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace cmsisnn {

/*! \brief Attributes to store the compiler options for CMSIS-NN. */
struct CMSISNNCompilerConfigNode : public tvm::AttrsNode<CMSISNNCompilerConfigNode> {
  String mcpu;
  String mattr;
  Bool debug_last_error = Bool(false);

  TVM_DECLARE_ATTRS(CMSISNNCompilerConfigNode, "ext.attrs.CMSISNNCompilerConfigNode") {
    TVM_ATTR_FIELD(mcpu)
        .describe(
            "The CPU to configure CMSIS-NN for (i.e. cortex-m55, cortex-m4), can also include "
            "attributes (i.e. cortex-m55+nomve)")
        .set_default("");
    TVM_ATTR_FIELD(mattr)
        .describe("The attributes to configure CMSIS-NN (i.e. +nodsp, +nomve)")
        .set_default("");
    TVM_ATTR_FIELD(debug_last_error)
        .describe("Whether to enable storing the last error")
        .set_default(Bool(false));
  }
};

class CMSISNNCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CMSISNNCompilerConfig, Attrs,
                                            CMSISNNCompilerConfigNode);
};

/*! \brief Convert External Code Generator options to TVM Target. */
Target CreateTarget(const tvm::transform::PassContext& ctx);

}  // namespace cmsisnn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_CMSISNN_COMPILER_ATTRS_H_
