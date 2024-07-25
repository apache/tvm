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
#ifndef TVM_SCRIPT_IR_BUILDER_RELAX_FRAME_H_
#define TVM_SCRIPT_IR_BUILDER_RELAX_FRAME_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/ir/frame.h>
#include <tvm/script/ir_builder/ir/ir.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

/*! \brief The base ir_builder frame for the relax dialect. */
class RelaxFrameNode : public IRBuilderFrameNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { IRBuilderFrameNode::VisitAttrs(v); }

  static constexpr const char* _type_key = "script.ir_builder.relax.RelaxFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(RelaxFrameNode, IRBuilderFrameNode);
};

class RelaxFrame : public IRBuilderFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RelaxFrame, IRBuilderFrame, RelaxFrameNode);

 protected:
  RelaxFrame() = default;
};

/*! \brief The base ir_builder frame for frames with SeqExpr
           i.e. Functions, If branches
  */
class SeqExprFrameNode : public RelaxFrameNode {
 public:
  /*! \brief The binding blocks inside the frame. */
  Array<tvm::relax::BindingBlock> binding_blocks;
  /*! \brief The frame output expr. `NullOpt` when undefined. */
  Optional<tvm::relax::Expr> output;

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("binding_blocks", &binding_blocks);
    v->Visit("output", &output);
  }

  static constexpr const char* _type_key = "script.ir_builder.relax.SeqExprFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(SeqExprFrameNode, RelaxFrameNode);

 public:
  void EnterWithScope() override;
  void ExitWithScope() override;
};

class SeqExprFrame : public RelaxFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(SeqExprFrame, RelaxFrame, SeqExprFrameNode);
};

/*! \brief The ir_builder frame for the relax function. */
class FunctionFrameNode : public SeqExprFrameNode {
 public:
  /*!
   * \brief The function name.
   * \note The name will not be specified in constructor, so it is "Optional",
   *       However, we must specify the name by `R.func_name` before exit this frame.
   */
  Optional<String> name;
  /*! \brief The function params. */
  Array<tvm::relax::Var> params;
  /*!
   * \brief The function return struct info.
   * \note Usually the function return type can be deduced by the function body.
   *       But we can use this field to specify a more "accurate" return type.
   *       i.e. If the `ret_struct_info` is None, try to use the deduced type from body
   *       If the `ret_struct_info` is not None, we can still take body.struct_info
   *       if we ret_struct_info is base of body.struct_info. If not, we will
   *       take the specified `ret_struct_info`.
   */
  Optional<tvm::relax::StructInfo> ret_struct_info;
  /*! \brief Whether the function is annotated as pure */
  Optional<Bool> is_pure;
  /*! \brief Whether the function is annotated as private */
  Optional<Bool> is_private;
  /*! \brief The function attributes. */
  Map<String, ObjectRef> attrs;
  /*! \brief The block builder to create Relax function. */
  tvm::relax::BlockBuilder block_builder;

  void VisitAttrs(tvm::AttrVisitor* v) {
    SeqExprFrameNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("params", &params);
    v->Visit("ret_struct_info", &ret_struct_info);
    v->Visit("is_pure", &is_pure);
    v->Visit("attrs", &attrs);
    v->Visit("binding_blocks", &binding_blocks);
    v->Visit("output", &output);
    // `block_builder` is not visited.
  }

  static constexpr const char* _type_key = "script.ir_builder.relax.FunctionFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionFrameNode, SeqExprFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class FunctionFrame : public SeqExprFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionFrame, SeqExprFrame, FunctionFrameNode);
};

/*! \brief The ir_builder frame for relax binding blocks. */
class BlockFrameNode : public RelaxFrameNode {
 public:
  /*! \brief The flag that indicates whether the block is a dataflow block. */
  bool is_dataflow;
  /*! \brief The variables emitted in this block. */
  Array<tvm::relax::Var> emitted_vars;
  /*!
   * \brief A boolean indicating if the dataflow block is ended of construction.
   * If it is true, any new binding trying to be emitted into this block will cause an error.
   * \note Only used for a dataflow block.
   */
  bool block_ended;
  /*!
   * \brief The output vars of the dataflow block.
   * \note Only used for a dataflow block.
   */
  Array<tvm::relax::Var> output_vars;

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("is_dataflow", &is_dataflow);
    v->Visit("emitted_vars", &emitted_vars);
    v->Visit("output_vars", &output_vars);
    // `block_ended` is not visited.
  }

  static constexpr const char* _type_key = "script.ir_builder.relax.BlockFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockFrameNode, RelaxFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class BlockFrame : public RelaxFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockFrame, RelaxFrame, BlockFrameNode);
};

/*!
 * \brief A frame that represents if statement.
 *
 * \sa IfFrame
 */
class IfFrameNode : public RelaxFrameNode {
 public:
  /*! \brief The condition of the if statement. */
  tvm::relax::Expr condition;
  /*! \brief The Bindings in the true branch. */
  Optional<tvm::relax::Expr> then_expr;
  /*! \brief The Bindings in the false branch. */
  Optional<tvm::relax::Expr> else_expr;
  /*! \brief The Binding var. */
  tvm::relax::Var var;
  /*! \brief The binding var name. */
  String var_name;

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
    v->Visit("then_expr", &then_expr);
    v->Visit("else_expr", &else_expr);
    v->Visit("var", &var);
    v->Visit("var_name", &var_name);
  }

  static constexpr const char* _type_key = "script.ir_builder.relax.IfFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfFrameNode, RelaxFrameNode);

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
 * \brief Managed reference to IfFrameNode.
 *
 * \sa IfFrameNode
 */
class IfFrame : public RelaxFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IfFrame, RelaxFrame, IfFrameNode);
};

/*!
 * \brief A frame that represents then.
 *
 * \sa ThenFrame
 */
class ThenFrameNode : public SeqExprFrameNode {
 public:
  static constexpr const char* _type_key = "script.ir_builder.relax.ThenFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThenFrameNode, SeqExprFrameNode);

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
class ThenFrame : public SeqExprFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ThenFrame, SeqExprFrame, ThenFrameNode);
};

/*!
 * \brief A frame that represents else.
 *
 * \sa ElseFrame
 */
class ElseFrameNode : public SeqExprFrameNode {
 public:
  static constexpr const char* _type_key = "script.ir_builder.relax.ElseFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ElseFrameNode, SeqExprFrameNode);

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
class ElseFrame : public SeqExprFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ElseFrame, SeqExprFrame, ElseFrameNode);
};

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_RELAX_FRAME_H_
