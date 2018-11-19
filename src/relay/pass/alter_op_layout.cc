/*!
 * Copyright (c) 2018 by Contributors
 * \file alter_op_layout.cc
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include "../op/layout.h"

namespace tvm {
namespace relay {

using LayoutMap = std::unordered_map<const Node*, LayoutInfo>;

class LayoutCorrector: public ExprMutator {
 public:
  LayoutCorrector() {

  }

  Expr Correct(Expr expr) {
    return expr;
  }
};

class LayoutAlternator: public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* n) {
    static auto falter_layout =
        Op::GetAttr<FTVMAlterOpLayout>("FTVMAlterOpLayout");

    Expr new_e = ExprMutator::VisitExpr_(n);
    const auto* new_n = new_e.as<CallNode>();

    if(!new_n->op.as<OpNode>())
      return new_e;

    Op op = Downcast<Op>(new_n->op);

    if (falter_layout.count(op)) {
      Expr ret = falter_layout[op](new_n->attrs, new_n->args);
      if (ret.defined())
        return ret;
    }
    return new_e;
  }
};

TVM_REGISTER_API("relay._ir_pass.AlterOpLayout")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Expr expr = args[0];
  LayoutCorrector corrector;

  expr = corrector.Correct(expr);
  expr = LayoutAlternator().Mutate(expr);
  expr = corrector.Correct(expr);

  *ret = expr;
});

}  // namespace relay
}  // namespace tvm
