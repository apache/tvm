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
#ifndef TVM_SCRIPT_IR_BUILDER_FRAME_H_
#define TVM_SCRIPT_IR_BUILDER_FRAME_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/stmt.h>
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
  ffi::Map<ffi::String, GlobalVar> global_var_map;
  /*!
   * \brief A map from GlobalVar to all global functions.
   * \note Only defined functions are in the map, while declared functions are not included.
   */
  ffi::Map<GlobalVar, BaseFunc> functions;
  /*! \brief IRModule's attributes. */
  ffi::Map<ffi::String, Any> attrs;
  /*! \brief IRModule's global_infos */
  ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IRModuleFrameNode>()
        .def_ro("global_vars", &IRModuleFrameNode::global_var_map)
        .def_ro("functions", &IRModuleFrameNode::functions)
        .def_ro("attrs", &IRModuleFrameNode::attrs)
        .def_ro("global_infos", &IRModuleFrameNode::global_infos);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.IRModuleFrame", IRModuleFrameNode,
                                    IRBuilderFrameNode);

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
  explicit IRModuleFrame(ffi::ObjectPtr<IRModuleFrameNode> data) : IRBuilderFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IRModuleFrame, IRBuilderFrame, IRModuleFrameNode);
};

/*! \brief A shared frame containing a sequence of core IR statements. */
class StmtFrameNode : public IRBuilderFrameNode {
 public:
  /*! \brief The statements within this frame. */
  ffi::Array<tvm::Stmt> stmts;

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<StmtFrameNode>().def_ro("stmts", &StmtFrameNode::stmts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.StmtFrame", StmtFrameNode, IRBuilderFrameNode);
};

/*! \brief Managed reference to StmtFrameNode. */
class StmtFrame : public IRBuilderFrame {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtFrame, IRBuilderFrame, StmtFrameNode);

 protected:
  StmtFrame() = default;
  explicit StmtFrame(ffi::ObjectPtr<StmtFrameNode> data) : IRBuilderFrame(data) {}
};

/*!
 * \brief A frame that represents the for loop.
 *
 * \sa ForFrame
 */
class ForFrameNode : public StmtFrameNode {
 public:
  /*!
   * \brief Functions that generate loop nests.
   * \param loop_vars The loop variables, from outer to inner
   * \param loop_extents The loop extents that correspond to loop variables
   * \param loop_body The loop body
   * \return A stmt, the loop nest
   */
  using FMakeForLoop = ffi::TypedFunction<tvm::Stmt(
      ffi::Array<tvm::Var> loop_vars, ffi::Array<Range> loop_extents,
      ffi::Array<ffi::Optional<PrimExpr>> loop_steps, tvm::SeqStmt loop_body, Span span)>;
  /*! \brief The loop variable. */
  ffi::Array<tvm::Var> vars;
  /*! \brief The domains of iteration. */
  ffi::Array<Range> doms;
  /*! \brief The optional steps of iteration. */
  ffi::Array<ffi::Optional<PrimExpr>> steps;
  /*! \brief The for loop generating function. */
  FMakeForLoop f_make_for_loop;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ForFrameNode>()
        .def_ro("vars", &ForFrameNode::vars)
        .def_ro("doms", &ForFrameNode::doms);
    // `f_make_for_loop` is not registered as it's not visited.
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.ForFrame", ForFrameNode, StmtFrameNode);

 public:
  /*! \brief Apply source target names before entry, preserving variable identity. */
  void SetNames(ffi::Optional<ffi::Variant<ffi::String, ffi::Array<ffi::String>>> names);
  /*! \brief Construct the loop nest with this frame's stored source location. */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to ForFrameNode.
 *
 * \sa ForFrameNode
 */
class ForFrame : public StmtFrame {
 public:
  explicit ForFrame(ffi::ObjectPtr<ForFrameNode> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ForFrame, StmtFrame, ForFrameNode);
};

/*!
 * \brief A frame that represents the assert statement. Proceeds if the condition is true,
 * otherwise aborts with the message.
 *
 * \sa AssertFrame
 */
class AssertFrameNode : public StmtFrameNode {
 public:
  explicit AssertFrameNode(ffi::UnsafeInit tag) : condition(tag), error_kind(tag) {}

  AssertFrameNode(PrimExpr condition, tvm::StringImm error_kind)
      : condition(std::move(condition)), error_kind(std::move(error_kind)) {}

  /*! \brief The PrimExpr to test. */
  PrimExpr condition;
  /*! \brief The error kind, e.g. "RuntimeError", "TypeError", "ValueError". */
  tvm::StringImm error_kind;
  /*! \brief Error message fragments, concatenated at runtime when assertion fails. */
  ffi::Array<tvm::StringImm> message_parts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AssertFrameNode>()
        .def_ro("condition", &AssertFrameNode::condition)
        .def_ro("error_kind", &AssertFrameNode::error_kind)
        .def_ro("message_parts", &AssertFrameNode::message_parts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.AssertFrame", AssertFrameNode,
                                    StmtFrameNode);

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
class AssertFrame : public StmtFrame {
 public:
  explicit AssertFrame(ffi::ObjectPtr<AssertFrameNode> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertFrame, StmtFrame, AssertFrameNode);
};

/*!
 * \brief A result-free region with lexical body parameters.
 * \sa RegionFrame
 */
class RegionFrameNode : public StmtFrameNode {
 public:
  explicit RegionFrameNode(ffi::UnsafeInit tag) : op(tag) {}
  explicit RegionFrameNode(Op op) : op(std::move(op)) {}

  /*! \brief The operation represented by this region. */
  Op op;
  /*! \brief Operands evaluated in the enclosing scope. */
  ffi::Array<Expr> args;
  /*! \brief Variables defined at entry to the region body. */
  ffi::Array<Var> body_params;
  /*! \brief Additional operation attributes. */
  DictAttrs attrs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RegionFrameNode>()
        .def_ro("op", &RegionFrameNode::op)
        .def_ro("args", &RegionFrameNode::args)
        .def_ro("body_params", &RegionFrameNode::body_params)
        .def_ro("attrs", &RegionFrameNode::attrs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.RegionFrame", RegionFrameNode,
                                    StmtFrameNode);

  /*! \brief Construct the region and add it to the enclosing builder frame. */
  void ExitWithScope() final;
};

/*! \brief Managed reference to RegionFrameNode. */
class RegionFrame : public StmtFrame {
 public:
  explicit RegionFrame(ffi::ObjectPtr<RegionFrameNode> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionFrame, StmtFrame, RegionFrameNode);
};

/*!
 * \brief A frame that represents while loop.
 *
 * \sa WhileFrame
 */
class WhileFrameNode : public StmtFrameNode {
 public:
  explicit WhileFrameNode(ffi::UnsafeInit tag) : condition(tag) {}

  explicit WhileFrameNode(PrimExpr condition) : condition(std::move(condition)) {}

  /*! \brief The termination condition of while. */
  PrimExpr condition;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WhileFrameNode>().def_ro("condition", &WhileFrameNode::condition);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.WhileFrame", WhileFrameNode,
                                    StmtFrameNode);

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
class WhileFrame : public StmtFrame {
 public:
  explicit WhileFrame(ffi::ObjectPtr<WhileFrameNode> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WhileFrame, StmtFrame, WhileFrameNode);
};

/*!
 * \brief A frame that represents if statement.
 *
 * \sa IfFrame
 */
class IfFrameNode : public StmtFrameNode {
 public:
  explicit IfFrameNode(ffi::UnsafeInit tag) : condition(tag) {}

  explicit IfFrameNode(PrimExpr condition) : condition(std::move(condition)) {}

  /*! \brief The condition of the if statement. */
  PrimExpr condition;
  /*! \brief The statements in the true branch. */
  ffi::Optional<ffi::Array<tvm::Stmt>> then_stmts;
  /*! \brief The stetements in the false branch. */
  ffi::Optional<ffi::Array<tvm::Stmt>> else_stmts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfFrameNode>()
        .def_ro("condition", &IfFrameNode::condition)
        .def_ro("then_stmts", &IfFrameNode::then_stmts)
        .def_ro("else_stmts", &IfFrameNode::else_stmts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.IfFrame", IfFrameNode, StmtFrameNode);

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
class IfFrame : public StmtFrame {
 public:
  explicit IfFrame(ffi::ObjectPtr<IfFrameNode> data) : StmtFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfFrame, StmtFrame, IfFrameNode);
};

/*!
 * \brief A frame that represents then.
 *
 * \sa ThenFrame
 */
class ThenFrameNode : public StmtFrameNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ThenFrameNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.ThenFrame", ThenFrameNode,
                                    StmtFrameNode);

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
class ThenFrame : public StmtFrame {
 public:
  explicit ThenFrame(ffi::ObjectPtr<ThenFrameNode> data) : StmtFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ThenFrame, StmtFrame, ThenFrameNode);
};

/*!
 * \brief A frame that represents else.
 *
 * \sa ElseFrame
 */
class ElseFrameNode : public StmtFrameNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ElseFrameNode>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.ElseFrame", ElseFrameNode,
                                    StmtFrameNode);

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
class ElseFrame : public StmtFrame {
 public:
  explicit ElseFrame(ffi::ObjectPtr<ElseFrameNode> data) : StmtFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ElseFrame, StmtFrame, ElseFrameNode);
};

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_FRAME_H_
