/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_subst.cc
 * \brief Function for substituting a concrete type in place of a type ID
 */
#include "./type_subst.h"
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

struct TypeSubstV : TypeMutator {
  tvm::Map<TypeVar, Type> subst_map;

  explicit TypeSubstV(tvm::Map<TypeVar, Type> subst_map)
    : subst_map(subst_map) {}

  Type VisitType_(const TypeVarNode* op) override {
    auto id = GetRef<TypeVar>(op);
    if (subst_map.find(id) != subst_map.end()) {
      return this->subst_map[id];
    } else {
      return id;
    }
  }
};

Type TypeSubst(const Type& type, const TypeVar& target, const Type& subst) {
  TypeSubstV ty_sub({ {target, subst} });
  return ty_sub.VisitType(type);
}

Type TypeSubst(const Type& type, tvm::Map<TypeVar, Type> subst_map) {
  TypeSubstV ty_sub(subst_map);
  return ty_sub.VisitType(type);
}

}  // namespace relay
}  // namespace tvm
