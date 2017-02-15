/*!
 *  Copyright (c) 2016 by Contributors
 *  (TODO)
 * \file api_arith.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/api_registry.h>
#include "../arithmetic/int_set.h"

namespace tvm {
namespace arith {

TVM_REGISTER_API(_arith_intset_single_point)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = IntSet::single_point(args[0]);
  });

TVM_REGISTER_API(_arith_intset_range)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = IntSet::range(args[0], args[1]);
  });

}  // namespace arith
}  // namespace tvm
