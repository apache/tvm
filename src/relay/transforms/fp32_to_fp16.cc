
#include "fp32_to_fp16.h"

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace relay {

using CallColorMap = std::unordered_map<const CallNode*, FP16ConversionCategory>;

class GraphColorer : private ExprVisitor {
  using ColorFunc = std::function<FP16ConversionCategory(const CallNode*)>;

 private:
  CallColorMap color_map;
  ColorFunc func;

  void VisitExpr_(const CallNode* l) final {
    // FP16ConversionCategory c = func(l);
    color_map[l] = func(l);
    ExprVisitor::VisitExpr_(l);
  }

 public:
  GraphColorer(ColorFunc func = DefaultColorer()) : func(func) {}

  CallColorMap result() { return color_map; }
  void VisitExpr(const Expr& expr) { ExprVisitor::VisitExpr(expr); }
};

class ColorPrinter : private ExprVisitor {
 private:
  CallColorMap color_map;

 public:
  explicit ColorPrinter(CallColorMap& color_map) : color_map(color_map) {}
  explicit ColorPrinter() {}
  void VisitExpr(const Expr& expr) { ExprVisitor::VisitExpr(expr); }

  void VisitExpr_(const CallNode* l) final {
    ExprVisitor::VisitExpr_(l);
    std::cout << l->op << " is " << conversion_category_strings[color_map[l]] << std::endl;
  }
};

void PrintColors(const Expr& expr) {
  GraphColorer initial_colorer = GraphColorer();
  initial_colorer.VisitExpr(expr);
  CallColorMap color_map = initial_colorer.result();
  ColorPrinter(color_map).VisitExpr(expr);
}
TVM_REGISTER_GLOBAL("relay._transform.PrintColorsExpr").set_body_typed(PrintColors);

}  // namespace relay
}  // namespace tvm
