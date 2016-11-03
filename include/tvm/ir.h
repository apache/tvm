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
}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_H_
