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
class IRModuleFrameObj : public IRBuilderFrameObj {
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
    refl::ObjectDef<IRModuleFrameObj>()
        .def_ro("global_vars", &IRModuleFrameObj::global_var_map)
        .def_ro("functions", &IRModuleFrameObj::functions)
        .def_ro("attrs", &IRModuleFrameObj::attrs)
        .def_ro("global_infos", &IRModuleFrameObj::global_infos);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.IRModuleFrame", IRModuleFrameObj,
                                    IRBuilderFrameObj);

 public:
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to IRModuleFrameObj.
 *
 * \sa IRModuleFrameObj
 */
class IRModuleFrame : public IRBuilderFrame {
 public:
  explicit IRModuleFrame(ffi::ObjectPtr<IRModuleFrameObj> data) : IRBuilderFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IRModuleFrame, IRBuilderFrame, IRModuleFrameObj);
};

/*! \brief A shared frame containing a sequence of core IR statements. */
class StmtFrameObj : public IRBuilderFrameObj {
 public:
  /*! \brief The statements within this frame. */
  ffi::Array<tvm::Stmt> stmts;

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<StmtFrameObj>().def_ro("stmts", &StmtFrameObj::stmts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("script.ir_builder.StmtFrame", StmtFrameObj, IRBuilderFrameObj);
};

/*! \brief Managed reference to StmtFrameObj. */
class StmtFrame : public IRBuilderFrame {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtFrame, IRBuilderFrame, StmtFrameObj);

 protected:
  StmtFrame() = default;
  explicit StmtFrame(ffi::ObjectPtr<StmtFrameObj> data) : IRBuilderFrame(data) {}
};

/*!
 * \brief A frame that represents the for loop.
 *
 * \sa ForFrame
 */
class ForFrameObj : public StmtFrameObj {
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
      ffi::Array<ffi::Optional<PrimExpr>> loop_steps, tvm::SeqStmt loop_body, Location loc)>;
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
    refl::ObjectDef<ForFrameObj>()
        .def_ro("vars", &ForFrameObj::vars)
        .def_ro("doms", &ForFrameObj::doms);
    // `f_make_for_loop` is not registered as it's not visited.
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.ForFrame", ForFrameObj, StmtFrameObj);

 public:
  /*! \brief Apply source target names before entry, preserving variable identity. */
  void SetNames(ffi::Optional<ffi::Variant<ffi::String, ffi::Array<ffi::String>>> names);
  /*! \brief Construct the loop nest with this frame's stored source location. */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to ForFrameObj.
 *
 * \sa ForFrameObj
 */
class ForFrame : public StmtFrame {
 public:
  explicit ForFrame(ffi::ObjectPtr<ForFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ForFrame, StmtFrame, ForFrameObj);
};

/*!
 * \brief A frame that represents the assert statement. Proceeds if the condition is true,
 * otherwise aborts with the message.
 *
 * \sa AssertFrame
 */
class AssertFrameObj : public StmtFrameObj {
 public:
  explicit AssertFrameObj(ffi::UnsafeInit tag) : condition(tag), error_kind(tag) {}

  AssertFrameObj(PrimExpr condition, tvm::StringImm error_kind)
      : condition(std::move(condition)), error_kind(std::move(error_kind)) {}

  /*! \brief The PrimExpr to test. */
  PrimExpr condition;
  /*! \brief The error kind, e.g. "RuntimeError", "TypeError", "ValueError". */
  tvm::StringImm error_kind;
  /*! \brief Error message fragments, concatenated at runtime when assertion fails. */
  ffi::Array<tvm::StringImm> message_parts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AssertFrameObj>()
        .def_ro("condition", &AssertFrameObj::condition)
        .def_ro("error_kind", &AssertFrameObj::error_kind)
        .def_ro("message_parts", &AssertFrameObj::message_parts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.AssertFrame", AssertFrameObj,
                                    StmtFrameObj);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to AssertFrameObj.
 *
 * \sa AssertFrameObj
 */
class AssertFrame : public StmtFrame {
 public:
  explicit AssertFrame(ffi::ObjectPtr<AssertFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(AssertFrame, StmtFrame, AssertFrameObj);
};

/*!
 * \brief A result-free region with lexical body parameters.
 * \sa RegionFrame
 */
class RegionFrameObj : public StmtFrameObj {
 public:
  explicit RegionFrameObj(ffi::UnsafeInit tag) : op(tag) {}
  explicit RegionFrameObj(Op op) : op(std::move(op)) {}

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
    refl::ObjectDef<RegionFrameObj>()
        .def_ro("op", &RegionFrameObj::op)
        .def_ro("args", &RegionFrameObj::args)
        .def_ro("body_params", &RegionFrameObj::body_params)
        .def_ro("attrs", &RegionFrameObj::attrs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.RegionFrame", RegionFrameObj,
                                    StmtFrameObj);

  /*! \brief Construct the region and add it to the enclosing builder frame. */
  void ExitWithScope() final;
};

/*! \brief Managed reference to RegionFrameObj. */
class RegionFrame : public StmtFrame {
 public:
  explicit RegionFrame(ffi::ObjectPtr<RegionFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RegionFrame, StmtFrame, RegionFrameObj);
};

/*!
 * \brief A frame that represents while loop.
 *
 * \sa WhileFrame
 */
class WhileFrameObj : public StmtFrameObj {
 public:
  explicit WhileFrameObj(ffi::UnsafeInit tag) : condition(tag) {}

  explicit WhileFrameObj(PrimExpr condition) : condition(std::move(condition)) {}

  /*! \brief The termination condition of while. */
  PrimExpr condition;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<WhileFrameObj>().def_ro("condition", &WhileFrameObj::condition);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.WhileFrame", WhileFrameObj,
                                    StmtFrameObj);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to WhileFrameObj.
 *
 * \sa WhileFrameObj
 */
class WhileFrame : public StmtFrame {
 public:
  explicit WhileFrame(ffi::ObjectPtr<WhileFrameObj> data) : StmtFrame(ffi::UnsafeInit{}) {
    TVM_FFI_ICHECK(data != nullptr);
    data_ = std::move(data);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(WhileFrame, StmtFrame, WhileFrameObj);
};

/*!
 * \brief A frame that represents if statement.
 *
 * \sa IfFrame
 */
class IfFrameObj : public StmtFrameObj {
 public:
  explicit IfFrameObj(ffi::UnsafeInit tag) : condition(tag) {}

  explicit IfFrameObj(PrimExpr condition) : condition(std::move(condition)) {}

  /*! \brief The condition of the if statement. */
  PrimExpr condition;
  /*! \brief The statements in the true branch. */
  ffi::Optional<ffi::Array<tvm::Stmt>> then_stmts;
  /*! \brief The stetements in the false branch. */
  ffi::Optional<ffi::Array<tvm::Stmt>> else_stmts;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<IfFrameObj>()
        .def_ro("condition", &IfFrameObj::condition)
        .def_ro("then_stmts", &IfFrameObj::then_stmts)
        .def_ro("else_stmts", &IfFrameObj::else_stmts);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.IfFrame", IfFrameObj, StmtFrameObj);

 public:
  /*!
   * \brief The method called when exiting RAII scope.
   * \sa tvm::support::With
   */
  void ExitWithScope() final;
};

/*!
 * \brief Managed reference to IfFrameObj.
 *
 * \sa IfFrameObj
 */
class IfFrame : public StmtFrame {
 public:
  explicit IfFrame(ffi::ObjectPtr<IfFrameObj> data) : StmtFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IfFrame, StmtFrame, IfFrameObj);
};

/*!
 * \brief A frame that represents then.
 *
 * \sa ThenFrame
 */
class ThenFrameObj : public StmtFrameObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ThenFrameObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.ThenFrame", ThenFrameObj, StmtFrameObj);

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
 * \brief Managed reference to ThenFrameObj.
 *
 * \sa ThenFrameObj
 */
class ThenFrame : public StmtFrame {
 public:
  explicit ThenFrame(ffi::ObjectPtr<ThenFrameObj> data) : StmtFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ThenFrame, StmtFrame, ThenFrameObj);
};

/*!
 * \brief A frame that represents else.
 *
 * \sa ElseFrame
 */
class ElseFrameObj : public StmtFrameObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ElseFrameObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.ir_builder.tirx.ElseFrame", ElseFrameObj, StmtFrameObj);

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
 * \brief Managed reference to ElseFrameObj.
 *
 * \sa ElseFrameObj
 */
class ElseFrame : public StmtFrame {
 public:
  explicit ElseFrame(ffi::ObjectPtr<ElseFrameObj> data) : StmtFrame(data) {
    TVM_FFI_ICHECK(data != nullptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ElseFrame, StmtFrame, ElseFrameObj);
};

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_FRAME_H_
