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
#ifndef TVM_TIRX_SCRIPT_IR_BUILDER_FRAME_H_
#define TVM_TIRX_SCRIPT_IR_BUILDER_FRAME_H_

#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/frame.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <functional>
#include <utility>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tirx {

// Retain dialect imports as aliases of the shared canonical frames.
using ir::AssertFrame;
using ir::AssertFrameNode;
using ir::ElseFrame;
using ir::ElseFrameNode;
using ir::ForFrame;
using ir::ForFrameNode;
using ir::IfFrame;
using ir::IfFrameNode;
using ir::RegionFrame;
using ir::RegionFrameNode;
using ir::ThenFrame;
using ir::ThenFrameNode;
using ir::WhileFrame;
using ir::WhileFrameNode;

/*!
 * \brief A statement frame with dialect-specific tensor alias policy.
 *
 * \sa TIRFrame
 */
class TIRFrameNode : public ir::StmtFrameNode {
 public:
  /*! \brief Bind a view in frames that support region aliases. */
  virtual void BindBufferRegion(tvm::tirx::TensorVar buffer, tvm::TensorRegion region);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TIRFrameNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.tirx.TIRFrame", TIRFrameNode, ir::StmtFrameNode);
};

/*!
 * \brief Managed reference to TIRFrameNode.
 *
 * \sa TIRFrameNode
 */
class TIRFrame : public ir::StmtFrame {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TIRFrame, ir::StmtFrame, TIRFrameNode);

 protected:
  TIRFrame() = default;
  explicit TIRFrame(ffi::ObjectPtr<TIRFrameNode> data) : ir::StmtFrame(data) {}
};

/*!
 * \brief A frame that represents the Function containing TIR statements.
 *
 * \sa FunctionFrame
 */
class FunctionFrameNode : public TIRFrameNode {
 public:
  /*! \brief The name of the block. */
  ffi::Optional<ffi::String> name;
  /*! \brief Function parameters. */
  ffi::Array<tvm::Var> args;
  /*! \brief Whether the Function is annotated as private. */
  bool is_private;
  /*! \brief The return type of the function. */
  ffi::Optional<Type> ret_type;
  /*! \brief Additional attributes storing the meta-data */
  ffi::Map<ffi::String, Any> attrs;
  /*! \brief Whether it is a persistent kernel. */
  bool persistent;
  /*! \brief Whether this frame declares a bodyless signature. */
  bool is_declaration{false};
  /*! \brief Finalized function and its module identity. */
  ffi::Optional<tvm::tirx::Function> function;
  ffi::Optional<GlobalVar> global_var;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FunctionFrameNode>()
        .def_ro("name", &FunctionFrameNode::name)
        .def_ro("args", &FunctionFrameNode::args)
        .def_ro("is_private", &FunctionFrameNode::is_private)
        .def_ro("ret_type", &FunctionFrameNode::ret_type)
        .def_ro("attrs", &FunctionFrameNode::attrs)
        .def_ro("persistent", &FunctionFrameNode::persistent)
        .def_ro("is_declaration", &FunctionFrameNode::is_declaration)
        .def_ro("function", &FunctionFrameNode::function)
        .def_ro("global_var", &FunctionFrameNode::global_var);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.tirx.FunctionFrame", FunctionFrameNode,
                              TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;

  /*! \brief Register validation for an extension-owned function attribute. */
  using AttrValidator = std::function<void(const FunctionFrameNode*, const ffi::Any&)>;
  static void RegisterAttrValidator(ffi::String key, AttrValidator validator);
  void ValidateAttrs() const;

  /*! \brief Complete dialect-specific function construction before publication. */
  virtual tvm::tirx::Function FinalizeFunction(tvm::tirx::Function func);
};

/*!
 * \brief Managed reference to FunctionFrameNode.
 *
 * \sa FunctionFrameNode
 */
class FunctionFrame : public TIRFrame {
 public:
  explicit FunctionFrame(ffi::ObjectPtr<FunctionFrameNode> data) : TIRFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(FunctionFrame, TIRFrame, FunctionFrameNode);
};

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_TIRX_SCRIPT_IR_BUILDER_FRAME_H_
