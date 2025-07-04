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
#ifndef TVM_SCRIPT_IR_BUILDER_TIR_FRAME_H_
#define TVM_SCRIPT_IR_BUILDER_TIR_FRAME_H_

#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/ir/frame.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

/*!
 * \brief A base frame that represents the TIR fame with body of statements.
 *
 * \sa TIRFrame
 */
class TIRFrameNode : public IRBuilderFrameNode {
 public:
  /*! \brief The Stmt within in this frame. */
  Array<tvm::tir::Stmt> stmts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TIRFrameNode>().def_ro("stmts", &TIRFrameNode::stmts);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.TIRFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRFrameNode, IRBuilderFrameNode);
};

/*!
 * \brief Managed reference to TIRFrameNode.
 *
 * \sa TIRFrameNode
 */
class TIRFrame : public IRBuilderFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, IRBuilderFrame, TIRFrameNode);

 protected:
  TIRFrame() = default;
};

/*!
 * \brief A frame that represents the PrimFunc containing TIR statements.
 *
 * \sa PrimFuncFrame
 */
class PrimFuncFrameNode : public TIRFrameNode {
 public:
  /*! \brief The name of the block. */
  Optional<String> name;
  /*! \brief Function parameters. */
  Array<tvm::tir::Var> args;
  /*! \brief Whether the PrimFunc is annotated as private. */
  bool is_private;
  /*! \brief The return type of the function. */
  Optional<Type> ret_type;
  /*! \brief Maps some parameters to specific Buffer data structures. */
  Map<tvm::tir::Var, tvm::tir::Buffer> buffer_map;
  /*! \brief Additional attributes storing the meta-data */
  Map<String, Any> attrs;
  /*! \brief The variable map bound to thread env. */
  Map<tvm::tir::Var, tvm::tir::IterVar> env_threads;
  /*! \brief The buffer allocated in root block. */
  Array<tvm::tir::Buffer> root_alloc_buffers;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrimFuncFrameNode>()
        .def_ro("name", &PrimFuncFrameNode::name)
        .def_ro("args", &PrimFuncFrameNode::args)
        .def_ro("is_private", &PrimFuncFrameNode::is_private)
        .def_ro("ret_type", &PrimFuncFrameNode::ret_type)
        .def_ro("buffer_map", &PrimFuncFrameNode::buffer_map)
        .def_ro("attrs", &PrimFuncFrameNode::attrs)
        .def_ro("env_threads", &PrimFuncFrameNode::env_threads)
        .def_ro("root_alloc_buffers", &PrimFuncFrameNode::root_alloc_buffers);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.PrimFuncFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimFuncFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to PrimFuncFrameNode.
 *
 * \sa PrimFuncFrameNode
 */
class PrimFuncFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrimFuncFrame, TIRFrame, PrimFuncFrameNode);
};

/*!
 * \brief A frame that represents the block.
 *
 * \sa BlockFrame
 */
class BlockFrameNode : public TIRFrameNode {
 public:
  /*! \brief The name of the block. */
  String name;
  /*! \brief The variables of the block. */
  Array<tvm::tir::IterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  Optional<Array<tvm::tir::BufferRegion>> reads;
  /*! \brief The write buffer regions of the block. */
  Optional<Array<tvm::tir::BufferRegion>> writes;
  /*! \brief The init statement of the bolck. */
  Optional<tvm::tir::Stmt> init;
  /*! \brief The buffer allocated in the block. */
  Array<tvm::tir::Buffer> alloc_buffers;
  /*! \brief The match buffer regions. */
  Array<tvm::tir::MatchBufferRegion> match_buffers;
  /*! \brief The annotation of the block. */
  Optional<Map<String, Any>> annotations;
  /*! \brief The corresponding values of the iter vars. */
  Array<PrimExpr> iter_values;
  /*!
   * \brief The predicate of the block realization, the block will only be executed when the
   * predicate is true.
   */
  Optional<PrimExpr> predicate;
  /*! \brief The flag whether to construct BlockRealize or Block. */
  bool no_realize;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BlockFrameNode>()
        .def_ro("name", &BlockFrameNode::name)
        .def_ro("iter_vars", &BlockFrameNode::iter_vars)
        .def_ro("reads", &BlockFrameNode::reads)
        .def_ro("writes", &BlockFrameNode::writes)
        .def_ro("init", &BlockFrameNode::init)
        .def_ro("alloc_buffers", &BlockFrameNode::alloc_buffers)
        .def_ro("match_buffers", &BlockFrameNode::match_buffers)
        .def_ro("annotations", &BlockFrameNode::annotations)
        .def_ro("iter_values", &BlockFrameNode::iter_values)
        .def_ro("predicate", &BlockFrameNode::predicate)
        .def_ro("no_realize", &BlockFrameNode::no_realize);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.BlockFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to BlockFrameNode.
 *
 * \sa BlockFrameNode
 */

class BlockFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockFrame, TIRFrame, BlockFrameNode);
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

  static constexpr const char* _type_key = "script.ir_builder.tir.BlockInitFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockInitFrameNode, TIRFrameNode);

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
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockInitFrame, TIRFrame, BlockInitFrameNode);
};

/*!
 * \brief A frame that represents the for loop.
 *
 * \sa ForFrame
 */
class ForFrameNode : public TIRFrameNode {
 public:
  /*!
   * \brief Functions that generate loop nests.
   * \param loop_vars The loop variables, from outer to inner
   * \param loop_extents The loop extents that correspond to loop variables
   * \param loop_body The loop body
   * \return A stmt, the loop nest
   */
  using FMakeForLoop = ffi::TypedFunction<tvm::tir::Stmt(
      Array<tvm::tir::Var> loop_vars, Array<Range> loop_extents, tvm::tir::Stmt loop_body)>;
  /*! \brief The loop variable. */
  Array<tvm::tir::Var> vars;
  /*! \brief The domains of iteration. */
  Array<Range> doms;
  /*! \brief The for loop generating function. */
  FMakeForLoop f_make_for_loop;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ForFrameNode>()
        .def_ro("vars", &ForFrameNode::vars)
        .def_ro("doms", &ForFrameNode::doms);
    // `f_make_for_loop` is not registered as it's not visited.
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.ForFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to ForFrameNode.
 *
 * \sa ForFrameNode
 */
class ForFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ForFrame, TIRFrame, ForFrameNode);
};

/*!
 * \brief A frame that represents the assert statement. Proceeds if the condition is true,
 * otherwise aborts with the message.
 *
 * \sa AssertFrame
 */
class AssertFrameNode : public TIRFrameNode {
 public:
  /*! \brief The PrimExpr to test. */
  PrimExpr condition;
  /*! \brief The output error message when the assertion failed. */
  PrimExpr message;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AssertFrameNode>()
        .def_ro("condition", &AssertFrameNode::condition)
        .def_ro("message", &AssertFrameNode::message);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AssertFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssertFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to AssertFrameNode.
 *
 * \sa AssertFrameNode
 */
class AssertFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AssertFrame, TIRFrame, AssertFrameNode);
};

/*!
 * \brief A frame represents the let binding expression, which binds a var.
 *
 * \sa LetFrameNode
 */
class LetFrameNode : public TIRFrameNode {
 public:
  /*! \brief The variable we bind to */
  tvm::tir::Var var;
  /*! \brief The value we bind var to */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LetFrameNode>()
        .def_ro("var", &LetFrameNode::var)
        .def_ro("value", &LetFrameNode::value);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.LetFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to LetFrameNode.
 *
 * \sa LetFrameNode
 */
class LetFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LetFrame, TIRFrame, LetFrameNode);
};

/*!
 * \brief The LaunchThreadFrameNode.
 * \note It is used only inside a PrimFunc.
 */
class LaunchThreadFrameNode : public TIRFrameNode {
 public:
  /*! \brief The extent of environment thread. */
  PrimExpr extent;
  /*! \brief The attribute key, could be either virtual_thread or thread_extent. */
  String attr_key;
  /*! \brief The iteration variable. */
  tvm::tir::IterVar iter_var;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<LaunchThreadFrameNode>()
        .def_ro("extent", &LaunchThreadFrameNode::extent)
        .def_ro("attr_key", &LaunchThreadFrameNode::attr_key)
        .def_ro("iter_var", &LaunchThreadFrameNode::iter_var);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.LaunchThreadFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(LaunchThreadFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to LaunchThreadFrameNode.
 *
 * \sa LaunchThreadFrameNode
 */
class LaunchThreadFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LaunchThreadFrame, TIRFrame,
                                                    LaunchThreadFrameNode);
};

/*!
 * \brief A frame that represents realization.
 *
 * \sa RealizeFrame
 */
class RealizeFrameNode : public TIRFrameNode {
 public:
  /*! \brief The region of buffer access. */
  tvm::tir::BufferRegion buffer_slice;
  /*! \brief The storage scope associated with this realization. */
  String storage_scope;
  /*! \brief The condition expression. */
  PrimExpr condition;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RealizeFrameNode>()
        .def_ro("buffer_slice", &RealizeFrameNode::buffer_slice)
        .def_ro("storage_scope", &RealizeFrameNode::storage_scope)
        .def_ro("condition", &RealizeFrameNode::condition);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.RealizeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(RealizeFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to RealizeFrameNode.
 *
 * \sa RealizeFrameNode
 */
class RealizeFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RealizeFrame, TIRFrame, RealizeFrameNode);
};

/*!
 * \brief A frame represents the allocate.
 *
 * \sa AllocateFrame
 */
class AllocateFrameNode : public TIRFrameNode {
 public:
  /*! \brief The extents of the allocate. */
  Array<PrimExpr> extents;
  /*! \brief The data type of the buffer. */
  DataType dtype;
  /*! \brief The storage scope. */
  String storage_scope;
  /*! \brief The condition. */
  PrimExpr condition;
  /*! \brief Additional annotation hints. */
  Map<String, Any> annotations;
  /*! \brief The buffer var. */
  tvm::tir::Var buffer_var;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllocateFrameNode>()
        .def_ro("extents", &AllocateFrameNode::extents)
        .def_ro("dtype", &AllocateFrameNode::dtype)
        .def_ro("storage_scope", &AllocateFrameNode::storage_scope)
        .def_ro("condition", &AllocateFrameNode::condition)
        .def_ro("annotations", &AllocateFrameNode::annotations)
        .def_ro("buffer_var", &AllocateFrameNode::buffer_var);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AllocateFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to AllocateFrameNode.
 *
 * \sa AllocateFrameNode
 */
class AllocateFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateFrame, TIRFrame, AllocateFrameNode);
};

/*!
 * \brief A frame represents the allocate constant.
 *
 * \sa AllocateConstFrame
 */
class AllocateConstFrameNode : public TIRFrameNode {
 public:
  /*! \brief The data type of the buffer. */
  DataType dtype;
  /*! \brief The extents of the allocate. */
  Array<PrimExpr> extents;
  /*! \brief The data associated with the constant. */
  tvm::runtime::NDArray data;
  /*! \brief The buffer var */
  tvm::tir::Var buffer_var;
  /*! \brief Additional annotations about the allocation. */
  Map<String, Any> annotations;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllocateConstFrameNode>()
        .def_ro("dtype", &AllocateConstFrameNode::dtype)
        .def_ro("extents", &AllocateConstFrameNode::extents)
        .def_ro("data", &AllocateConstFrameNode::data)
        .def_ro("buffer_var", &AllocateConstFrameNode::buffer_var)
        .def_ro("annotations", &AllocateConstFrameNode::annotations);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AllocateConstFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateConstFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to AllocateConstFrameNode.
 *
 * \sa AllocateConstFrameNode
 */
class AllocateConstFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateConstFrame, TIRFrame,
                                                    AllocateConstFrameNode);
};
/*!
 * \brief A frame that represents attribute node.
 *
 * \sa AttrFrame
 */
class AttrFrameNode : public TIRFrameNode {
 public:
  /*! \brief The node to annotate the attribute. */
  ObjectRef node;
  /*! \brief Attribute type key. */
  String attr_key;
  /*! \brief The value of the attribute. */
  PrimExpr value;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AttrFrameNode>()
        .def_ro("node", &AttrFrameNode::node)
        .def_ro("attr_key", &AttrFrameNode::attr_key)
        .def_ro("value", &AttrFrameNode::value);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AttrFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to AttrFrameNode.
 *
 * \sa AttrFrameNode
 */
class AttrFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AttrFrame, TIRFrame, AttrFrameNode);
};

/*!
 * \brief A frame that represents while loop.
 *
 * \sa WhileFrame
 */
class WhileFrameNode : public TIRFrameNode {
 public:
  /*! \brief The termination condition of while. */
  PrimExpr condition;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WhileFrameNode>().def_ro("condition", &WhileFrameNode::condition);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.WhileFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(WhileFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to WhileFrameNode.
 *
 * \sa WhileFrameNode
 */
class WhileFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(WhileFrame, TIRFrame, WhileFrameNode);
};

/*!
 * \brief A frame that represents if statement.
 *
 * \sa IfFrame
 */
class IfFrameNode : public TIRFrameNode {
 public:
  /*! \brief The condition of the if statement. */
  PrimExpr condition;
  /*! \brief The statements in the true branch. */
  Optional<Array<tvm::tir::Stmt>> then_stmts;
  /*! \brief The stetements in the false branch. */
  Optional<Array<tvm::tir::Stmt>> else_stmts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfFrameNode>()
        .def_ro("condition", &IfFrameNode::condition)
        .def_ro("then_stmts", &IfFrameNode::then_stmts)
        .def_ro("else_stmts", &IfFrameNode::else_stmts);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.IfFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfFrameNode, TIRFrameNode);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to IfFrameNode.
 *
 * \sa IfFrameNode
 */
class IfFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IfFrame, TIRFrame, IfFrameNode);
};

/*!
 * \brief A frame that represents then.
 *
 * \sa ThenFrame
 */
class ThenFrameNode : public TIRFrameNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ThenFrameNode>();
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.ThenFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThenFrameNode, TIRFrameNode);

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
 * \brief Managed reference to ThenFrameNode.
 *
 * \sa ThenFrameNode
 */
class ThenFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ThenFrame, TIRFrame, ThenFrameNode);
};

/*!
 * \brief A frame that represents else.
 *
 * \sa ElseFrame
 */
class ElseFrameNode : public TIRFrameNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ElseFrameNode>();
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.ElseFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ElseFrameNode, TIRFrameNode);

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
 * \brief Managed reference to ElseFrameNode.
 *
 * \sa ElseFrameNode
 */
class ElseFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ElseFrame, TIRFrame, ElseFrameNode);
};

class DeclBufferFrameNode : public TIRFrameNode {
 public:
  /*! \brief The declared buffer. */
  tvm::tir::Buffer buffer;
  /*! \brief The buffer allocated or not. */
  bool allocated;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DeclBufferFrameNode>()
        .def_ro("buffer", &DeclBufferFrameNode::buffer)
        .def_ro("allocated", &DeclBufferFrameNode::allocated);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.DeclBufferFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(DeclBufferFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class DeclBufferFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(DeclBufferFrame, TIRFrame, DeclBufferFrameNode);
};

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_TIR_FRAME_H_
