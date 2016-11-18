#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/ir_mutator.h>

namespace {
using namespace tvm::ir;
using namespace Halide::Internal;
using namespace Halide;

// replace variable to constant
class IRVar2Const : public IRMutator {
 public:
  VarExpr var;
  int int_val;
  Expr mutate(Expr expr) final {
    static const FMutateExpr& f = IRVar2Const::vtable_expr();
    return (f.can_dispatch(expr) ?
            f(expr, expr, this) : IRMutator::mutate(expr));
  }
  static FMutateExpr &vtable_expr();
};

// implement vtable
IRMutator::FMutateExpr &IRVar2Const::vtable_expr() {  // NOLINT(*)
  static FMutateExpr inst; return inst;
}

TVM_STATIC_IR_FUNCTOR(IRVar2Const, vtable_expr)
.set_dispatch<Variable>([](const Variable* op, const Expr &e, IRMutator* m) {
    IRVar2Const* vm = static_cast<IRVar2Const*>(m);
    if (e.same_as(vm->var)) {
      return IntImm::make(Int(32), vm->int_val);
    } else {
      return e;
    }
  });

}  // namespace

TEST(IRMutator, Basic) {
  using namespace Halide::Internal;
  using namespace tvm;
  Var x("x"), y;
  auto z = x + y;
  IRVar2Const mu;
  mu.var = y;
  mu.int_val = 10;
  auto zz = mu.mutate(z);
  std::ostringstream os;
  os << zz;
  CHECK(os.str() == "(x + 10)");
}

TEST(IRMutator, Substitute) {
  using namespace Halide::Internal;
  using namespace tvm;
  Var x("x"), y;
  auto z = x + y;
  {
    auto zz = Substitute({{y.get(), 11}}, z);
    std::ostringstream os;
    os << zz;
    CHECK(os.str() == "(x + 11)");
  }
  {
    auto zz = Substitute({{z.get(), 11}}, z);
    std::ostringstream os;
    os << zz;
    CHECK(os.str() == "11");
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
