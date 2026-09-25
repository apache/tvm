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
#ifndef TVM_S_TIR_SCRIPT_BUILDER_FRAME_H_
#define TVM_S_TIR_SCRIPT_BUILDER_FRAME_H_

#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/script/builder/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace s_tir {

using tirx::TIRFrame;
using tirx::TIRFrameNode;

/*! \brief Function frame owning schedulable TIR construction. */
class PrimFuncFrameNode : public tirx::PrimFuncFrameNode {
 public:
  /*! \brief Buffers allocated in the implicit root block. */
  ffi::Array<tvm::tirx::BufferVar> root_alloc_buffers;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimFuncFrameNode>().def_ro("root_alloc_buffers",
                                                &PrimFuncFrameNode::root_alloc_buffers);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.s_tir.PrimFuncFrame", PrimFuncFrameNode,
                                    tirx::PrimFuncFrameNode);
  tvm::tirx::PrimFunc FinalizeFunction(tvm::tirx::PrimFunc func) final;
};

class PrimFuncFrame : public tirx::PrimFuncFrame {
 public:
  explicit PrimFuncFrame(ffi::ObjectPtr<PrimFuncFrameNode> data)
      : tirx::PrimFuncFrame(std::move(data)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrimFuncFrame, tirx::PrimFuncFrame,
                                                PrimFuncFrameNode);
};

/*!
 * \brief A frame that represents the block.
 *
 * \sa SBlockFrame
 */
class SBlockFrameNode : public TIRFrameNode {
 public:
  /*! \brief The name of the block. */
  ffi::String name;
  /*! \brief The variables of the block. */
  ffi::Array<tvm::tirx::IterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  ffi::Optional<ffi::Array<tvm::TensorRegion>> reads;
  /*! \brief The write buffer regions of the block. */
  ffi::Optional<ffi::Array<tvm::TensorRegion>> writes;
  /*! \brief The init statement of the bolck. */
  ffi::Optional<tvm::tirx::Stmt> init;
  /*! \brief The buffer allocated in the block. */
  ffi::Array<tvm::tirx::BufferVar> alloc_buffers;
  /*! \brief The match buffer regions. */
  ffi::Array<tvm::s_tir::MatchBufferRegion> match_buffers;
  /*! \brief The annotation of the block. */
  ffi::Optional<ffi::Map<ffi::String, Any>> annotations;
  /*! \brief The corresponding values of the iter vars. */
  ffi::Array<PrimExpr> iter_values;
  /*!
   * \brief The predicate of the block realization, the block will only be executed when the
   * predicate is true.
   */
  ffi::Optional<PrimExpr> predicate;
  /*! \brief The flag whether to construct BlockRealize or Block. */
  bool no_realize;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockFrameNode>()
        .def_ro("name", &SBlockFrameNode::name)
        .def_ro("iter_vars", &SBlockFrameNode::iter_vars)
        .def_ro("reads", &SBlockFrameNode::reads)
        .def_ro("writes", &SBlockFrameNode::writes)
        .def_ro("init", &SBlockFrameNode::init)
        .def_ro("alloc_buffers", &SBlockFrameNode::alloc_buffers)
        .def_ro("match_buffers", &SBlockFrameNode::match_buffers)
        .def_ro("annotations", &SBlockFrameNode::annotations)
        .def_ro("iter_values", &SBlockFrameNode::iter_values)
        .def_ro("predicate", &SBlockFrameNode::predicate)
        .def_ro("no_realize", &SBlockFrameNode::no_realize);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.s_tir.SBlockFrame", SBlockFrameNode,
                                    TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void BindBufferRegion(tvm::tirx::BufferVar buffer, tvm::TensorRegion region) final;
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to SBlockFrameNode.
 *
 * \sa SBlockFrameNode
 */

class SBlockFrame : public TIRFrame {
 public:
  explicit SBlockFrame(ffi::ObjectPtr<SBlockFrameNode> data) : TIRFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlockFrame, TIRFrame, SBlockFrameNode);
};

/*!
 * \brief A frame that represents the block initialization statment.
 *
 * \sa BlockInitFrame
 */
class BlockInitFrameNode : public TIRFrameNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BlockInitFrameNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.s_tir.BlockInitFrame", BlockInitFrameNode,
                                    TIRFrameNode);

 public:
  /*!
   * \brief The method called when entering RAII scope.
   * \sa tvm::support::With
   */
  void EnterWithScope() final;
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to BlockInitFrameNode.
 *
 * \sa BlockInitFrameNode
 */
class BlockInitFrame : public TIRFrame {
 public:
  explicit BlockInitFrame(ffi::ObjectPtr<BlockInitFrameNode> data) : TIRFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(BlockInitFrame, TIRFrame, BlockInitFrameNode);
};

}  // namespace s_tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_S_TIR_SCRIPT_BUILDER_FRAME_H_
