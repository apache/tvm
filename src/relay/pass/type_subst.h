/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/pass/type_subst.h
 * \brief Utility functions for substituting types.
 */
#ifndef TVM_RELAY_PASS_TYPE_SUBST_H_
#define TVM_RELAY_PASS_TYPE_SUBST_H_

#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

Type TypeSubst(const Type& type, const TypeParam& target, const Type& subst);
Type TypeSubst(const Type& type, tvm::Map<TypeParam, Type> subst_map);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPE_SUBST_H_
