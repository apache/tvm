/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions related to Codegen
 * \file c_api_codegen.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/codegen.h>
#include <tvm/lowered_func.h>
#include <tvm/api_registry.h>

#include "../codegen/codegen_hybrid.h"

namespace tvm {
namespace codegen {

TVM_REGISTER_API("codegen._Build")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<LoweredFunc>()) {
      *ret = Build({args[0]}, args[1]);
    } else {
      *ret = Build(args[0], args[1]);
    }
  });

TVM_REGISTER_API("module._PackImportsToC")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = PackImportsToC(args[0], args[1]);
  });

TVM_REGISTER_API("codegen._HybridDump")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Stmt stmt;
    if (args[0].IsNodeType<Stmt>()) {
      stmt = args[0];
    } else if (args[0].IsNodeType<Expr>()) {
      stmt = Evaluate::make(args[0]);
    }
    CodeGenHybrid generator;
    generator.PrintStmt(stmt);
    *ret = generator.Finish();
  });
}  // namespace codegen
}  // namespace tvm
