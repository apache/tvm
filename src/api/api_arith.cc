/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to arith
 * \file api_arith.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/api_registry.h>
#include <tvm/tensor.h>

namespace tvm {
namespace arith {

TVM_REGISTER_API("arith.intset_single_point")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = IntSet::single_point(args[0]);
  });

TVM_REGISTER_API("arith.intset_vector")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = IntSet::vector(args[0]);
  });

TVM_REGISTER_API("arith.intset_interval")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = IntSet::interval(args[0], args[1]);
  });

TVM_REGISTER_API("arith.DetectLinearEquation")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = DetectLinearEquation(args[0], args[1]);
  });

TVM_REGISTER_API("arith.DetectClipBound")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = DetectClipBound(args[0], args[1]);
  });

TVM_REGISTER_API("arith.DeduceBound")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = DeduceBound(args[0], args[1],
        args[2].operator Map<Var, IntSet>(),
        args[3].operator Map<Var, IntSet>());
  });


TVM_REGISTER_API("arith.DomainTouched")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = DomainTouched(args[0], args[1], args[2], args[3]);
  });


TVM_REGISTER_API("_IntervalSetGetMin")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator IntSet().min();
  });

TVM_REGISTER_API("_IntervalSetGetMax")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator IntSet().max();
  });

TVM_REGISTER_API("_IntSetIsNothing")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator IntSet().is_nothing();
  });

TVM_REGISTER_API("_IntSetIsEverything")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator IntSet().is_everything();
  });

TVM_REGISTER_API("arith._make_ConstIntBound")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = ConstIntBoundNode::make(args[0], args[1]);
  });

TVM_REGISTER_API("arith._make_ModularSet")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = ModularSetNode::make(args[0], args[1]);
  });

TVM_REGISTER_API("arith._CreateAnalyzer")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    using runtime::PackedFunc;
    using runtime::TypedPackedFunc;
    auto self = std::make_shared<Analyzer>();
    auto f = [self](std::string name) -> PackedFunc {
      if (name == "const_int_bound") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->const_int_bound(args[0]);
          });
      } else if (name == "modular_set") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->modular_set(args[0]);
        });
      } else if (name == "const_int_bound_update") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            self->const_int_bound.Update(args[0], args[1], args[2]);
        });
      } else if (name == "bind") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            auto& sptr = args[1].node_sptr();
            if (sptr->is_type<Range::ContainerType>()) {
              self->Bind(args[0], args[1].operator Range());
            } else {
              self->Bind(args[0], args[1].operator Expr());
            }
        });
      } else if (name == "enter_constraint_context") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            // can't use make_shared due to noexcept(false) decl in destructor,
            // see https://stackoverflow.com/a/43907314
            auto ctx =
                std::shared_ptr<ConstraintContext>(new ConstraintContext(self.get(), args[0]));
            auto fexit = [ctx](TVMArgs, TVMRetValue*) mutable {
              ctx.reset();
            };
            *ret = PackedFunc(fexit);
        });
      }
      return PackedFunc();
    };
    *ret = TypedPackedFunc<PackedFunc(std::string)>(f);
});

}  // namespace arith
}  // namespace tvm
