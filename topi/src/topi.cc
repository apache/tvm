/*!
*  Copyright (c) 2017 by Contributors
* \brief Registration of TVM operators and schedules
* \file topi.cc
*/
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>

#include <topi/broadcast.h>

namespace topi {

using namespace tvm;
using namespace tvm::runtime;

TVM_REGISTER_GLOBAL("topi.broadcast_add")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_add(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_sub")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_sub(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_mul")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_mul(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_div")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_div(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_maximum")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_maximum(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("topi.broadcast_minimum")
.set_body([](TVMArgs args, TVMRetValue *rv) {
  *rv = broadcast_minimum(args[0], args[1]);
   });

}  // namespace topi
