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
#ifndef TVM_SCRIPT_IR_BUILDER_IR_FRAME_H_
#define TVM_SCRIPT_IR_BUILDER_IR_FRAME_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/node/node.h>
#include <tvm/script/ir_builder/base.h>

#include <vector>

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

/*!
 * \brief A frame that represents the IRModule frame with functions and global variables.
 *
 * \sa IRModuleFrame
 */
class IRModuleFrameNode : public IRBuilderFrameNode {
 public:
  /*! \brief A map from string names to global variables that ensures global uniqueness. */
  Map<String, GlobalVar> global_var_map;
  /*!
   * \brief A map from GlobalVar to all global functions.
   * \note Only defined functions are in the map, while declared functions are not included.
   */
  Map<GlobalVar, BaseFunc> functions;
  /*! \brief IRModule's attributes. */
  Map<String, ObjectRef> attrs;
  /*! \brief IRModule's global_infos */
  Map<String, Array<GlobalInfo>> global_infos;

  void VisitAttrs(tvm::AttrVisitor* v) {
    IRBuilderFrameNode::VisitAttrs(v);
    v->Visit("global_vars", &global_var_map);
    v->Visit("functions", &functions);
    v->Visit("attrs", &attrs);
    v->Visit("global_infos", &global_infos);
  }

  static constexpr const char* _type_key = "script.ir_builder.IRModuleFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRModuleFrameNode, IRBuilderFrameNode);

 public:
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to IRModuleFrameNode.
 *
 * \sa IRModuleFrameNode
 */
class IRModuleFrame : public IRBuilderFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRModuleFrame, IRBuilderFrame,
                                                    IRModuleFrameNode);
};

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_IR_FRAME_H_
