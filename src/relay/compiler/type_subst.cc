/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_subst.cc
 * \brief Function for substituting a concrete type in place of a type ID
 */
#include "./type_subst.h"
#include "./type_visitor.h"

namespace tvm {
namespace relay {

struct TypeSubst : TypeFVisitor {
  tvm::Map<TypeParam, Type> subst_map;

  explicit TypeSubst(tvm::Map<TypeParam, Type> subst_map)
    : subst_map(subst_map) {}

  Type VisitType_(const TypeParamNode *op) override {
    auto id = GetRef<TypeParam>(op);
    if (subst_map.find(id) != subst_map.end()) {
      return this->subst_map[id];
    } else {
      return id;
    }
  }
};

Type type_subst(const Type &type, const TypeParam &target, const Type &subst) {
  TypeSubst ty_sub({ {target, subst} });
  return ty_sub.VisitType(type);
}

Type type_subst(const Type &type, tvm::Map<TypeParam, Type> subst_map) {
  TypeSubst ty_sub(subst_map);
  return ty_sub.VisitType(type);
}

}  // namespace relay
}  // namespace tvm
