/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir.h
 * \brief Additional high level nodes in the IR
 */
#ifndef TVM_IR_H_
#define TVM_IR_H_

#include <ir/Expr.h>
#include <ir/IR.h>
#include <type_traits>
#include <string>
#include "./base.h"
#include "./domain.h"

namespace tvm {
namespace ir {

using Halide::Internal::ExprNode;
using Halide::Internal::IRNodeType;
using Halide::Internal::ForType;

/*! \brief Reduction operator operator */
struct Reduce : public ExprNode<Reduce> {
  /*!
   * \brief The binary operator of reduction
   */
  std::string op;
  /*! \brief The source operand */
  Expr source;
  /*! \brief The reduction domain */
  RDomain rdom;

  /*! \brief construct expr from name and rdom */
  static Expr make(std::string name, Expr src, RDomain rdom);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("op", &op);
    v->Visit("source", &source);
    v->Visit("rdom", &rdom);
  }
  static const IRNodeType _type_info = IRNodeType::ExtensionExpr;
  static constexpr const char* _type_key = "Reduce";
  static constexpr const char* Add = "Add";
  static constexpr const char* Max = "Max";
  static constexpr const char* Min = "Min";
};

// Reuse IR node defintiion from HalideIR
using Halide::Internal::IntImm;
using Halide::Internal::UIntImm;
using Halide::Internal::FloatImm;
using Halide::Internal::StringImm;
using Halide::Internal::Cast;
using Halide::Internal::Variable;
using Halide::Internal::Add;
using Halide::Internal::Sub;
using Halide::Internal::Mul;
using Halide::Internal::Div;
using Halide::Internal::Mod;
using Halide::Internal::Min;
using Halide::Internal::Max;
using Halide::Internal::EQ;
using Halide::Internal::NE;
using Halide::Internal::LT;
using Halide::Internal::LE;
using Halide::Internal::GT;
using Halide::Internal::GE;
using Halide::Internal::And;
using Halide::Internal::Or;
using Halide::Internal::Not;
using Halide::Internal::Select;
using Halide::Internal::Load;
using Halide::Internal::Ramp;
using Halide::Internal::Broadcast;
using Halide::Internal::Call;
using Halide::Internal::Let;
using Halide::Internal::LetStmt;
using Halide::Internal::AssertStmt;
using Halide::Internal::ProducerConsumer;
using Halide::Internal::For;
using Halide::Internal::Store;
using Halide::Internal::Provide;
using Halide::Internal::Allocate;
using Halide::Internal::Free;
using Halide::Internal::Realize;
using Halide::Internal::Block;
using Halide::Internal::IfThenElse;
using Halide::Internal::Evaluate;

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_H_
