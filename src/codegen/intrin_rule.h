/*!
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule.h
 * \brief Utility to generate intrinsic rules
 */
#ifndef TVM_CODEGEN_INTRIN_RULE_H_
#define TVM_CODEGEN_INTRIN_RULE_H_

#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/api_registry.h>
#include <tvm/runtime/registry.h>
#include <string>

namespace tvm {
namespace codegen {
namespace intrin {
using namespace ir;

// Add float suffix to the intrinsics
struct FloatSuffix {
  std::string operator()(Type t, std::string name) const {
    if (t == Float(32)) {
      return name + 'f';
    } else if (t == Float(64)) {
      return name;
    } else {
      return "";
    }
  }
};

// Return the intrinsic name
struct Direct {
  std::string operator()(Type t, std::string name) const {
    return name;
  }
};

// Call pure extern function.
template<typename T>
inline void DispatchExtern(const TVMArgs& args, TVMRetValue* rv) {
  Expr e = args[0];
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  std::string name = T()(call->type, call->name);
  if (name.length() != 0) {
    *rv = Call::make(
        call->type, name, call->args, Call::PureExtern);
  } else {
    *rv = e;
  }
}

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_INTRIN_RULE_H_
