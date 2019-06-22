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
.set_body_typed(IntSet::single_point);

TVM_REGISTER_API("arith.intset_vector")
.set_body_typed(IntSet::vector);

TVM_REGISTER_API("arith.intset_interval")
.set_body_typed(IntSet::interval);


TVM_REGISTER_API("arith.DetectLinearEquation")
.set_body_typed(DetectLinearEquation);

TVM_REGISTER_API("arith.DetectClipBound")
.set_body_typed(DetectClipBound);

TVM_REGISTER_API("arith.DeduceBound")
.set_body_typed<IntSet(Expr, Expr, Map<Var, IntSet>, Map<Var, IntSet>)>([](
  Expr v, Expr cond,
  const Map<Var, IntSet> hint_map,
  const Map<Var, IntSet> relax_map
) {
  return DeduceBound(v, cond, hint_map, relax_map);
});


TVM_REGISTER_API("arith.DomainTouched")
.set_body_typed(DomainTouched);

TVM_REGISTER_API("_IntervalSetGetMin")
.set_body_method(&IntSet::min);

TVM_REGISTER_API("_IntervalSetGetMax")
.set_body_method(&IntSet::max);

TVM_REGISTER_API("_IntSetIsNothing")
.set_body_method(&IntSet::is_nothing);

TVM_REGISTER_API("_IntSetIsEverything")
.set_body_method(&IntSet::is_everything);

ConstIntBound MakeConstIntBound(int64_t min_value, int64_t max_value) {
  return ConstIntBound(min_value, max_value);
}

TVM_REGISTER_API("arith._make_ConstIntBound")
.set_body_typed(MakeConstIntBound);

ModularSet MakeModularSet(int64_t coeff, int64_t base) {
  return ModularSet(coeff, base);
}

TVM_REGISTER_API("arith._make_ModularSet")
.set_body_typed(MakeModularSet);

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
      } else if (name == "rewrite_simplify") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->rewrite_simplify(args[0]);
        });
      } else if (name == "canonical_simplify") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->canonical_simplify(args[0]);
        });
      } else if (name == "int_set") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->int_set(args[0], args[1]);
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
            auto ctx = std::shared_ptr<With<ConstraintContext> >(
                new With<ConstraintContext>(self.get(), args[0]));
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
