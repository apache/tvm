#include <iostream>
#include <string>
#include <vector>

#include "../tvm/src/codegen/codegen_c.h"
#include "tvm/ir.h"
#include "tvm/tvm.h"
#include "tvm/ir_pass.h"
#include "tvm/schedule_pass.h"

namespace tvm{
namespace ir{
class MyExprFunctor:public tvm::ir::ExprFunctor<int(const Expr&,int)>{
        public:
                int VisitExpr_(const Variable*op, int b)final{
                return b;
        }
                int VisitExpr_(const IntImm*op, int b) final{
                return op->value;
        }
        int VisitExpr_(const Add *op, int b)final{
                return VisitExpr(op->a,b)+VisitExpr(op->b,b);
                //return b;
        }
};
}
}

using namespace std;
class tvm::ir::MyExprFunctor;

int main(void) {

#if 1
	tvm::ir::MyExprFunctor f;
	tvm::Var x("x");
	CHECK_EQ(f(x+1,2),3);
#endif

}
