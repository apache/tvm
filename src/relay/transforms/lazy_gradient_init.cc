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
 *
 * \file lazy_gradient_init.cc
 *
 * \brief Lazily instantiate 0-filled or 1-filled tensors.
 * This pass should be used after reverse-mode ad so that gradient tensors
 * are not instantiated until after the forward pass.
 *
 * This pass delays or removes memory allocation by converting tensors into
 * GradCell, an algebraic data type defined in gradient.rly.
 *
 * This will delay or decrease memory usage. All calls to
 * ones, ones_like, zeros, zeros_like will call the One or Zero constructor
 * of GradCell, which will not instantiate in memory until needed. All other cases result
 * in using the Raw constructor which means the tensor is instantiated in memory.
 *
 * It also overloads + and * operation which can increase performance when doing
 * operations involving tensors with values of only 0 or 1.
 *
 * Note: this pass can only be used with functions where the input/output types are a
 * combination of TupleTypes, TensorTypes, ADTs, and non-nested FuncTypes
 *
 * This pass optimizes 6 ops:
 * - add
 * - multiply
 * - ones
 * - ones_like
 * - zeros
 * - zeros_like
 *
 * This module level pass adds a new "GradCell" version datatype for each existing datatype.
 * This is the case to propogate the new GradCell datatype through ADTs such as Lists.
 * For each function, a new function is created that accepts the "GradCell" type of the arguments
 * of the original function. That is, inputs to the function are converted to their
 * GradCell-version, passed to the newly created "GradCell_Function". The output is then necessarily
 * converted from the GradCell version to the original return type.
 *
 * To support ADTs, we use functions that convert between an instance of an ADT to its
 * respective GradCell version
 * by matching constructors to the constructor of the "GradCell" datatype.
 *
 * A transformation function is required for different type arguments.
 * For example the ADT List may be List[int32] or List[List[int32]], which should be handled
 * separately.
 *
 * This pass uses 4 primary mutators:
 * - LazyGradientInitializer to create the "GradCell_Function" of a given function.
 * - GradCellWrapper mutates expr into its respective GradCell expr
 * - GradCellWrapper mutates expr into its respective non-GradCell expr
 * - ADTTransform creates a ADT for each unique ADT
 */

#include <tvm/ir/type_functor.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/transform.h>

#include <string>

#include "let_list.h"

namespace tvm {
namespace relay {

// prefix of name of GradCell version ADT
const char GradCell_Header[] = "_GradCell_";
// prefix of transformation function for converting ADT to GradCell version
const char GradCell_TransFunc[] = "_GradCell_TransFunc_";
// prefix of transformation function for converting GradCell version ADT to normal
const char GradCell_ReverseTransFunc[] = "_GradCell_ReverseTransFunc_";
// prefix of copy of function that operates on GradCell types
const char GradCell_Func[] = "_GradCell_Func_";

struct TypeCallHash {
  size_t operator()(const TypeCall& typecall) const { return ObjectHash()(typecall->func); }
};

/*!
 * \brief Check if two ADT instances are equal,
 * check for dataflow equivalence allow for mapping between TypeVars
 * i.e GradCell[TypeVar(A)] = GradCell[TypeVar(B)]
 */
struct TypeCallEqual {
  bool operator()(const TypeCall& l, const TypeCall& r) const {
    if (!(l->func.same_as(r->func))) {
      return false;
    }

    if (l->args.size() != r->args.size()) {
      return false;
    }

    for (size_t i = 0; i < l->args.size(); i++) {
      if (!tvm::StructuralEqual()(l->args[i], r->args[i], true)) {
        return false;
      }
    }

    return true;
  }
};

/*!
 * \brief ADTTransform creates a new ADT named
 * GradCell_Header + name_hint for each unique ADT.
 */
class ADTTransform : public TypeMutator, public PatternMutator {
 public:
  explicit ADTTransform(IRModule module) : module_(module) {}

  Type VisitType(const Type& t) final { return TypeMutator::VisitType(t); }

  Type VisitType_(const TensorTypeNode* op) final {
    GlobalTypeVar gradCell = module_->GetGlobalTypeVar("GradCell");
    tvm::Array<Type> args;
    args.push_back(GetRef<TensorType>(op));
    return TypeCall(gradCell, args);
  }

  Type VisitType_(const GlobalTypeVarNode* op) final {
    GlobalTypeVar t = GetRef<GlobalTypeVar>(op);
    if (op->kind == kAdtHandle) {
      if (adt_mapping_.count(t) != 0) {
        return adt_mapping_.at(t);
      }

      TypeData adt = module_->LookupTypeDef(t);
      this->VisitType(adt);

      return adt_mapping_.at(t);
    }

    return GetRef<Type>(op);
  }

  Type VisitType_(const TypeDataNode* op) final {
    auto type_data = GetRef<TypeData>(op);
    std::string transformed_adt_name = GradCell_Header + op->header->name_hint;

    // add new ADT to map to handle recursive definitions
    GlobalTypeVar new_adt = GlobalTypeVar(transformed_adt_name, op->header->kind);
    adt_mapping_[type_data->header] = new_adt;
    reverse_adt_mapping_[new_adt] = type_data->header;

    // define transformed ADT
    Array<Constructor> constructors;
    for (Constructor con : op->constructors) {
      Array<Type> inputs;
      for (Type t : con->inputs) {
        inputs.push_back(this->VisitType(t));
      }
      Constructor transformed_cons = Constructor(GradCell_Header + con->name_hint, inputs, new_adt);
      constructors.push_back(transformed_cons);
    }

    TypeData new_datatype = TypeData(new_adt, op->type_vars, constructors);
    module_->AddTypeDef(new_adt, new_datatype);
    return new_datatype;
  }

  Pattern VisitPattern(const Pattern& c) final { return PatternMutator::VisitPattern(c); }

  Constructor VisitConstructor(const Constructor& c) final {
    this->VisitType(c->belong_to);
    return module_->GetConstructor(GradCell_Header + c->belong_to->name_hint,
                                   GradCell_Header + c->name_hint);
  }

  /*!
   * \brief Given a transformed ADT, returned the original ADT.
   * Useful for GradCellUnWrapper which needs to map transformed ADT constructors
   * to the original ADT constructors.
   *
   * \param transformed_adt_handle GlobalTypeVar of "GradCell-version" of ADT
   * \return ADT
   */
  GlobalTypeVar GetReverseADT(GlobalTypeVar transformed_adt_handle) {
    auto it = reverse_adt_mapping_.find(transformed_adt_handle);

    // reverse mapping should always be found
    CHECK(it != reverse_adt_mapping_.end()) << "Reverse mapping of ADT transformation not found";
    return it->second;
  }

 private:
  // Module
  IRModule module_;
  // ADT -> transformed ADT
  std::unordered_map<GlobalTypeVar, GlobalTypeVar, ObjectHash, ObjectEqual> adt_mapping_;
  // transformed ADT -> ADT
  std::unordered_map<GlobalTypeVar, GlobalTypeVar, ObjectHash, ObjectEqual> reverse_adt_mapping_;
};

/*!
 * \brief Helper for TypeCallMutator.
 * Replace TypeVar with type arguments
 */
class TypeVarSolver : public TypeMutator {
 public:
  explicit TypeVarSolver(
      const std::unordered_map<TypeVar, TypeVar, ObjectHash, ObjectEqual>& type_var_map,
      const std::unordered_map<TypeVar, Type, ObjectHash, ObjectEqual>& type_call_map)
      : type_var_map_(type_var_map), type_call_map_(type_call_map) {}
  Type VisitType_(const TypeVarNode* op) final {
    TypeVar type = GetRef<TypeVar>(op);

    if (type_call_map_.count(type) != 0) {
      // recursively visit Type argument to replace possible nested TypeVar
      return VisitType(type_call_map_.at(type));
    }

    if (type_var_map_.count(type) != 0) {
      return type_var_map_.at(type);
    }

    return type;
  }

 private:
  // type vars to unique type vars
  std::unordered_map<TypeVar, TypeVar, ObjectHash, ObjectEqual> type_var_map_;
  // TypeCall arguments to ADT
  std::unordered_map<TypeVar, Type, ObjectHash, ObjectEqual> type_call_map_;
};

/*!
 * \brief Find all TypeVars within the arguments of a TypeCallNode and create a mapping
 * of the TypeVars to new TypeVars
 */
class TypeCallMutator : public TypeVisitor {
 public:
  // TypeVars within TypeCallNode
  Array<Type> args;
  // unique TypeVars
  Array<TypeVar> params;
  explicit TypeCallMutator(IRModule module, const TypeCallNode* op) : module_(module) {
    for (Type t : op->args) {
      // visit each type argument
      VisitType(t);
    }
    for (auto const& x : type_var_map) {
      args.push_back(x.first);
      params.push_back(x.second);
    }
  }

  /*!
   * \brief Replace ADT type vars with TypeCall arguments
   * and replace type vars with unique typevars
   *
   * \param t TypeCall
   * \param map TypeVar of ADT -> type argument
   *
   * \return type after replacing ADT TypeVar with arguments and replacing any
   * free type vars with uniquely generated typevars
   */

  Type InputType(Type t, const std::unordered_map<TypeVar, Type, ObjectHash, ObjectEqual>& map) {
    return TypeVarSolver(type_var_map, map).VisitType(t);
  }

  void VisitType_(const TypeVarNode* op) final {
    TypeVar tv = GetRef<TypeVar>(op);
    if (type_var_map.count(tv) == 0) {
      TypeVar replacement = TypeVar(tv->name_hint + "_", tv->kind);
      type_var_map.insert({tv, replacement});
    }
  }

 private:
  IRModule module_;
  // TypeVar in argument -> TypeVar of polymorphic function
  std::unordered_map<TypeVar, TypeVar, ObjectHash, ObjectEqual> type_var_map;
};

typedef class GradCellUnWrapper GradCellUnWrapper;

/*!
 * \brief Mutate a given expression into its "GradCell-version".
 * TensorTypes are wrapped with the Raw constructor of GradCell.
 * TupleTypes are recursively visited.
 * ADTTypes are converted to its appropriate transformed ADT
 * FuncTypes are wrapped with a function that appropriately wraps/unwraps input and output
 */
class GradCellWrapper : public ExprFunctor<Expr(const Expr&, const Type&, GradCellUnWrapper*)>,
                        public TypeMutator {
 public:
  explicit GradCellWrapper(IRModule module, ADTTransform* adt_transformer)
      : module_(module), adt_transformer_(adt_transformer), unique(0) {}
  Expr VisitExpr_(const VarNode* op, const Type& t, GradCellUnWrapper* unwrapper) final;
  Expr VisitExpr_(const TupleGetItemNode* op, const Type& t, GradCellUnWrapper* unwrapper) final;
  Expr VisitExpr_(const CallNode* op, const Type& t, GradCellUnWrapper* unwrapper) final;

 private:
  // Module
  IRModule module_;
  // ADTTransform
  ADTTransform* adt_transformer_;
  // TypeCall -> Function to transform an ADT Instance into GradCell version
  std::unordered_map<TypeCall, GlobalVar, TypeCallHash, TypeCallEqual> adt_wrapper_map_;
  // TypeVar of ADT call -> Type argument
  std::unordered_map<TypeVar, Type, ObjectHash, ObjectEqual> type_var_map;
  // append to prefix to create unique function names for ADT wrapper functions
  int64_t unique;

  Expr WrapExpr(const Expr expr, const Type& type, GradCellUnWrapper* unwrapper);
  // Return function to wrap ADT
  Expr GetADTFunction(const TypeCallNode* op, TypeCallMutator& type_args,
                      GradCellUnWrapper* unwrapper);
  Type VisitType_(const GlobalTypeVarNode* op) final;
  Type VisitType_(const TensorTypeNode* op) final;
};

/*!
 * \brief Mutate a given "GradCell-version" expression into its nonGradCell-version.
 * TypeCalls to GradCell are wrapped with FromGradCell function
 * TupleTypes are recursively visited.
 * Transformed ADTs are converted to its appropriate normal ADT
 */
class GradCellUnWrapper : public ExprFunctor<Expr(const Expr&, const Type&)>, public TypeMutator {
 public:
  explicit GradCellUnWrapper(IRModule module, ADTTransform* adt_transformer)
      : module_(module), adt_transformer_(adt_transformer), unique(0) {}
  Expr VisitExpr_(const VarNode* op, const Type& t) final;
  Expr VisitExpr_(const TupleGetItemNode* op, const Type& t) final;
  Expr VisitExpr_(const CallNode* op, const Type& t) final;
  Expr VisitExpr_(const TupleNode* op, const Type& t) final;
  Expr VisitExpr_(const ConstantNode* op, const Type& t) final;

 private:
  // Module
  IRModule module_;
  // ADTTransform
  ADTTransform* adt_transformer_;
  // TypeCall -> Function an GradCell_ADT into ADT
  std::unordered_map<TypeCall, GlobalVar, TypeCallHash, TypeCallEqual> adt_unwrapper_map_;
  // TypeVar of GradCell_ADT call -> Type argument
  std::unordered_map<TypeVar, Type, ObjectHash, ObjectEqual> type_var_map;
  // create unique strings for ADT unwrapper functions
  int64_t unique;

  Expr UnwrapExpr(const Expr expr, const Type& type);
  // Return function to unwrap ADT
  Expr GetReverseADTFunction(const TypeCallNode* op, TypeCallMutator& type_args);
  Type VisitType_(const TypeCallNode* op) final;
  Type VisitType_(const GlobalTypeVarNode* op) final;
};

/* GradCellWrapper */
Expr GradCellWrapper::VisitExpr_(const VarNode* op, const Type& t, GradCellUnWrapper* unwrapper) {
  return WrapExpr(GetRef<Var>(op), op->type_annotation, unwrapper);
}

Expr GradCellWrapper::VisitExpr_(const TupleGetItemNode* op, const Type& t,
                                 GradCellUnWrapper* unwrapper) {
  return WrapExpr(GetRef<TupleGetItem>(op), t, unwrapper);
}

Expr GradCellWrapper::VisitExpr_(const CallNode* op, const Type& t, GradCellUnWrapper* unwrapper) {
  return WrapExpr(GetRef<Call>(op), t, unwrapper);
}

Expr GradCellWrapper::WrapExpr(const Expr expr, const Type& type, GradCellUnWrapper* unwrapper) {
  if (type.as<TensorTypeNode>()) {
    return Call(module_->GetConstructor("GradCell", "Raw"), {expr}, Attrs(), {type});
  }

  if (auto* type_anno = type.as<TupleTypeNode>()) {
    tvm::Array<Expr> fields;
    for (size_t i = 0; i < type_anno->fields.size(); i++) {
      const Type& t = type_anno->fields[i];
      // recursively visit each item of tuple
      fields.push_back(this->VisitExpr(TupleGetItem(expr, i), t, unwrapper));
    }
    Expr tuple = Tuple(fields);
    return tuple;
  }

  if (auto* type_anno = type.as<TypeCallNode>()) {
    // create GradCell_ADT if not already created
    adt_transformer_->VisitType(type_anno->func);
    // find all type vars within type_anno
    // to handle polymorphic functions
    auto tvs = TypeCallMutator(module_, type_anno);

    return Call(GetADTFunction(type_anno, tvs, unwrapper), {expr}, Attrs(), tvs.args);
  }

  if (auto* type_anno = type.as<FuncTypeNode>()) {
    // to handle functions, we need to create a new function
    // that handles GradCell version input and outputs GradCell version types
    Array<Var> funcVars;
    Array<Expr> args;
    for (Type t : type_anno->arg_types) {
      Type visited = this->VisitType(t);
      Var v = Var("v", visited);
      funcVars.push_back(v);
      // unwrap arguments
      args.push_back(unwrapper->VisitExpr(v, visited));
    }
    // call original expr with unwrapped arguments
    Call call = Call(expr, args);
    // wrap results of the call
    Expr result = this->WrapExpr(call, type_anno->ret_type, unwrapper);
    // return new function with GradCell-version types, wrapping original function
    return Function(funcVars, result, this->VisitType(type_anno->ret_type), type_anno->type_params);
  }

  return expr;
}

Expr GradCellWrapper::GetADTFunction(const TypeCallNode* op, TypeCallMutator& type_args,
                                     GradCellUnWrapper* unwrapper) {
  auto type = GetRef<TypeCall>(op);
  GlobalTypeVar adt_handle = Downcast<GlobalTypeVar>(op->func);
  if (adt_wrapper_map_.count(type) != 0) {
    // ADT already wrapped previously
    return adt_wrapper_map_.at(type);
  }

  // handle recursive ADT which require recursive calls to transform
  GlobalVar func_var = GlobalVar(GradCell_Header + std::string(GradCell_TransFunc) +
                                 adt_handle->name_hint + std::to_string(unique++));
  adt_wrapper_map_[type] = func_var;

  TypeData adt_data = module_->LookupTypeDef(adt_handle);
  TypeData new_adt_data = module_->LookupTypeDef(GradCell_Header + adt_handle->name_hint);

  // solve for input type to wrap ADT function
  for (size_t i = 0; i < adt_data->type_vars.size(); i++) {
    type_var_map[adt_data->type_vars[i]] = op->args[i];
  }
  auto input_type = type_args.InputType(type, type_var_map);

  CHECK(adt_data->constructors.size() == new_adt_data->constructors.size())
      << "ADT and transformed ADT have different number of constructors";

  /*
   * Pattern match each constructor of the ADT to the respective constructor
   * in the transformed ADT. PatternVars then need to be recursively wrapped,
   * and passed as argument to the constructor of the transformed ADT
   */
  Array<Clause> clauses;
  for (size_t i = 0; i < adt_data->constructors.size(); i++) {
    // get Constructor to pattern match against
    Array<Pattern> patternVars;
    Array<Expr> c_args;
    Constructor c = adt_data->constructors[i];
    for (Type t : c->inputs) {
      // solve for type of PatternVar
      Type pattern_var_type = type_args.InputType(t, type_var_map);
      Var v = Var("var", pattern_var_type);
      patternVars.push_back(PatternVar(v));
      // recursively wrap
      c_args.push_back(this->VisitExpr(v, pattern_var_type, unwrapper));
    }
    Pattern p = PatternConstructor(c, patternVars);
    // return Constructor of new ADT with wrapped arguments
    Expr e = Call(new_adt_data->constructors[i], c_args);

    clauses.push_back(Clause(p, e));
  }

  Var v = Var("v", input_type);
  Expr match = Match(v, clauses);

  Function func = Function({v}, match, this->VisitType(input_type), type_args.params);
  module_->AddUnchecked(func_var, func);
  return func;
}

Type GradCellWrapper::VisitType_(const GlobalTypeVarNode* op) {
  GlobalTypeVar t = GetRef<GlobalTypeVar>(op);
  if (op->kind == kAdtHandle) {
    return adt_transformer_->VisitType(t);
  }

  return GetRef<Type>(op);
}

Type GradCellWrapper::VisitType_(const TensorTypeNode* op) {
  GlobalTypeVar gradCell = module_->GetGlobalTypeVar("GradCell");
  tvm::Array<Type> args;
  args.push_back(GetRef<TensorType>(op));
  return TypeCall(gradCell, args);
}

/* GradCellUnWrapper */
Expr GradCellUnWrapper::VisitExpr_(const CallNode* op, const Type& t) {
  return UnwrapExpr(GetRef<Call>(op), t);
}

Expr GradCellUnWrapper::VisitExpr_(const TupleGetItemNode* op, const Type& t) {
  return UnwrapExpr(GetRef<TupleGetItem>(op), t);
}

Expr GradCellUnWrapper::VisitExpr_(const VarNode* op, const Type& t) {
  return UnwrapExpr(GetRef<Var>(op), op->type_annotation);
}

Expr GradCellUnWrapper::VisitExpr_(const TupleNode* op, const Type& t) {
  return UnwrapExpr(GetRef<Tuple>(op), t);
}

Expr GradCellUnWrapper::VisitExpr_(const ConstantNode* op, const Type& t) {
  return UnwrapExpr(GetRef<Constant>(op), t);
}

Expr GradCellUnWrapper::UnwrapExpr(const Expr expr, const Type& type) {
  if (auto* type_call = type.as<TypeCallNode>()) {
    if (type_call->func.same_as(module_->GetGlobalTypeVar("GradCell"))) {
      // if TypeCall to GradCell, simply wrap with FromGradCell function
      return Call(module_->GetGlobalVar("FromGradCell"), {expr}, Attrs(), type_call->args);
    }

    // convert transformed ADT to ADT
    auto tvs = TypeCallMutator(module_, type_call);
    return Call(GetReverseADTFunction(type_call, tvs), {expr}, Attrs(), tvs.args);
  }

  if (auto* type_anno = type.as<TupleTypeNode>()) {
    tvm::Array<Expr> fields;
    for (size_t i = 0; i < type_anno->fields.size(); i++) {
      // recursively unwrap items of tuple
      const Type& t = type_anno->fields[i];
      fields.push_back(this->VisitExpr(TupleGetItem(expr, i), t));
    }
    Expr tuple = Tuple(fields);
    return tuple;
  }
  return expr;
}

Expr GradCellUnWrapper::GetReverseADTFunction(const TypeCallNode* op, TypeCallMutator& type_args) {
  TypeCall type = GetRef<TypeCall>(op);
  GlobalTypeVar transformed_adt_handle = Downcast<GlobalTypeVar>(op->func);
  GlobalTypeVar adt_handle = adt_transformer_->GetReverseADT(transformed_adt_handle);

  // sanity check
  CHECK(std::string(transformed_adt_handle->name_hint).rfind(GradCell_Header, 0) == 0)
      << "Output ADT is not a transformed ADT";

  if (adt_unwrapper_map_.count(type)) {
    // transformed ADT unwrapped previously
    return adt_unwrapper_map_.at(type);
  }

  // handle recursive ADTs
  GlobalVar func_var = GlobalVar(GradCell_Header + std::string(GradCell_ReverseTransFunc) +
                                 adt_handle->name_hint + std::to_string(unique++));
  adt_unwrapper_map_[type] = func_var;

  TypeData adt_data = module_->LookupTypeDef(adt_handle);
  TypeData transformed_adt_data = module_->LookupTypeDef(transformed_adt_handle);

  CHECK(adt_data->type_vars.size() == transformed_adt_data->type_vars.size())
      << "ADT and transformed ADT have different # of type args";

  // solve for TypeVars of ADT to solve for input type of function
  for (size_t i = 0; i < transformed_adt_data->type_vars.size(); i++) {
    type_var_map[adt_data->type_vars[i]] = op->args[i];
  }
  auto input_type = type_args.InputType(type, type_var_map);

  CHECK(adt_data->constructors.size() == transformed_adt_data->constructors.size())
      << "ADT and transformed ADT have different number of constructors";

  // use same logic as wrapping expression
  // Pattern match with each Constructor of the transformed ADT,
  // return respective Constructor with arguments of unwrapped PatternVars
  Array<Clause> clauses;
  for (size_t i = 0; i < transformed_adt_data->constructors.size(); i++) {
    // Get Constructor of transformed ADT
    Array<Pattern> patternVars;
    Array<Expr> c_args;
    Constructor c = transformed_adt_data->constructors[i];
    for (Type t : c->inputs) {
      // solve for type of pattern var
      Type pattern_var_type = type_args.InputType(t, type_var_map);
      Var v = Var("var", pattern_var_type);
      // bind PatternVar to Var passed to constructor
      patternVars.push_back(PatternVar(v));
      // recursively unwrap
      c_args.push_back(this->VisitExpr(v, pattern_var_type));
    }
    Pattern p = PatternConstructor(c, patternVars);
    // Call appropriate Constructor
    Expr e = Call(adt_data->constructors[i], c_args);

    clauses.push_back(Clause(p, e));
  }

  Var v = Var("v", input_type);
  Expr match = Match(v, clauses);

  Function func = Function({v}, match, this->VisitType(input_type), type_args.params);
  module_->AddUnchecked(func_var, func);
  return func;
}

Type GradCellUnWrapper::VisitType_(const TypeCallNode* op) {
  if (op->func.same_as(module_->GetGlobalTypeVar("GradCell"))) {
    return op->args[0];
  }
  return TypeMutator::VisitType_(op);
}

Type GradCellUnWrapper::VisitType_(const GlobalTypeVarNode* op) {
  GlobalTypeVar t = GetRef<GlobalTypeVar>(op);
  if (op->kind == kAdtHandle) {
    return adt_transformer_->GetReverseADT(t);
  }

  return GetRef<Type>(op);
}

class LazyGradientInitializer : public ExprMutator, public TypeMutator, public PatternMutator {
 public:
  explicit LazyGradientInitializer(IRModule module) : module_(module) {
    // setup
    adt_transformer_ = new ADTTransform(module_);
    grad_cell_wrapper_ = new GradCellWrapper(module_, adt_transformer_);
    grad_cell_unwrapper_ = new GradCellUnWrapper(module_, adt_transformer_);

    // import GradCell and GradCell functions
    module_->ImportFromStd("gradient.rly");

    // ignore these functions when transforming
    GlobalVar from_grad_cell = module_->GetGlobalVar("FromGradCell");
    GlobalVar mul_grad_cell = module_->GetGlobalVar("MultiplyGradCell");
    GlobalVar add_grad_cell = module_->GetGlobalVar("AddGradCell");

    func_map_[from_grad_cell] = from_grad_cell;
    func_map_[mul_grad_cell] = mul_grad_cell;
    func_map_[add_grad_cell] = add_grad_cell;
  }

  /*!
   * \brief Given a global function, create new global function
   * that mirrors the functionality however using GradCell type.
   * Original function will wrap inputs, call the mirrored function, unwrap the ouput,
   * and return.
   */
  BaseFunc VisitGlobalVar(const GlobalVar& gv) {
    auto base_func = module_->Lookup(gv);
    if (auto* e = base_func.as<FunctionNode>()) {
      auto f = GetRef<Function>(e);
      if (func_map_.count(gv) == 0) {
        // create GlobalVar handle for function
        func_map_[gv] = GlobalVar(GradCell_Func + gv->name_hint);
      }
      GlobalVar func_var = func_map_.at(gv);
      if (module_->ContainGlobalVar(func_var->name_hint)) {
        // transformed function already contained in IRModule, return
        return module_->Lookup(func_var);
      }
      // create transformed function and add definition to IRModule
      auto* transformed = ExprMutator::Mutate(f).as<FunctionNode>();
      module_->AddUnchecked(func_var, GetRef<Function>(transformed));

      // wrap inputs of Tensor type using GradCellWrapper class
      tvm::Array<Expr> args;
      for (Var var : f->params) {
        Expr wrappedInput =
            grad_cell_wrapper_->VisitExpr(var, var->checked_type(), grad_cell_unwrapper_);
        args.push_back(wrappedInput);
      }
      Expr transformedExpr = Call(func_var, args);

      // unwrap outputs of GradCell type into Tensor type using OutputVisitor class
      Expr tensorOutput = grad_cell_unwrapper_->VisitExpr(transformedExpr, transformed->ret_type);
      return Function(f->params, tensorOutput, f->ret_type, f->type_params);
    }
    LOG(FATAL) << "GlobalVar does not map to a function";
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    return Call(module_->GetConstructor("GradCell", "Raw"), {GetRef<Constant>(op)}, Attrs(),
                {op->checked_type()});
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    if (auto* op = (call_node->op).as<OpNode>()) {
      Expr op_expr = GetRef<Op>(op);

      if (op_expr == Op::Get("add")) {
        return CallGradCellFunction(call_node, module_->GetGlobalVar("AddGradCell"));
      }

      if (op_expr == Op::Get("multiply")) {
        return CallGradCellFunction(call_node, module_->GetGlobalVar("MultiplyGradCell"));
      }

      if (op_expr == Op::Get("ones") || op_expr == Op::Get("zeros")) {
        // ones and zeros need TensorType input
        Expr result = CallPrimitiveOp(call_node);
        Expr func = Function({}, result, {call_node->checked_type()}, Array<TypeVar>());
        // call appropriate GradCell constructor
        std::string constructor_name = op_expr == Op::Get("ones") ? "One" : "Zero";
        return Call(module_->GetConstructor("GradCell", constructor_name), {func}, Attrs(),
                    {call_node->checked_type()});
      }

      if (op_expr == Op::Get("ones_like") || op_expr == Op::Get("zeros_like")) {
        // ones_like and zeros_like need TensorType input
        Expr result = CallPrimitiveOp(call_node);
        // fn() -> T, function returns result of operation
        Expr func = Function({}, result, {call_node->checked_type()}, Array<TypeVar>());
        // call appropriate GradCell constructor
        std::string constructor_name = op_expr == Op::Get("ones_like") ? "One" : "Zero";
        return Call(module_->GetConstructor("GradCell", "One"), {func}, Attrs(),
                    {call_node->checked_type()});
      }

      // handle all other ops
      Expr result = CallPrimitiveOp(call_node);
      // wrap result with Raw constructor
      return grad_cell_wrapper_->VisitExpr(result, call_node->checked_type(), grad_cell_unwrapper_);
    }

    if (auto* op = (call_node->op).as<ConstructorNode>()) {
      // create "GradCell-version" of ADT if not already created
      adt_transformer_->VisitType(op->belong_to);
      // call Constructor of transformed ADT
      Constructor c = module_->GetConstructor(GradCell_Header + op->belong_to->name_hint,
                                              GradCell_Header + op->name_hint);
      Array<Expr> args;
      for (Expr e : call_node->args) {
        args.push_back(this->VisitExpr(e));
      }

      Array<Type> type_args;
      for (Type t : call_node->type_args) {
        type_args.push_back(this->VisitType(t));
      }
      return Call(c, args, Attrs(), type_args);
    }

    return ExprMutator::VisitExpr_(call_node);
  }

  Expr VisitExpr_(const ConstructorNode* op) final {
    Constructor c = module_->GetConstructor(GradCell_Header + op->belong_to->name_hint,
                                            GradCell_Header + op->name_hint);
    return c;
  }

  Expr VisitExpr_(const IfNode* op) final {
    auto true_b = VisitExpr(op->true_branch);
    auto false_b = VisitExpr(op->false_branch);

    // guard is bool type which will become GradCell[bool], so necessary to unwrap
    auto guard =
        grad_cell_unwrapper_->VisitExpr(VisitExpr(op->cond), VisitType(op->cond->checked_type()));
    return If(guard, true_b, false_b);
  }

  Expr VisitExpr_(const VarNode* op) final {
    auto var = GetRef<Var>(op);
    if (var_map_.count(var) != 0) {
      return var_map_.at(var);
    }

    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const GlobalVarNode* op) final {
    // GlobalVar is a handle to a global function
    GlobalVar gv = GetRef<GlobalVar>(op);
    if (func_map_.count(gv) == 0) {
      // create handle to transformed function
      func_map_[gv] = GlobalVar(GradCell_Func + op->name_hint);
    }
    return func_map_.at(gv);
  }

  Type VisitType(const Type& t) final { return TypeMutator::VisitType(t); }

  Type VisitType_(const GlobalTypeVarNode* op) final {
    GlobalTypeVar t = GetRef<GlobalTypeVar>(op);
    if (module_->GetGlobalTypeVar("GradCell").same_as(t)) {
      // if GradCell type, do nothing
      return t;
    }
    if (op->kind == kAdtHandle) {
      // handle to ADT, define GradCell version of ADT is not already created
      return adt_transformer_->VisitType(t);
    }

    return t;
  }

  Var VisitVar(const Var& v) final {
    // used for PatternMutator
    if (var_map_.count(v) == 0) {
      var_map_.insert(std::pair<Var, Var>(v, Var(v->name_hint(), VisitType(v->type_annotation))));
    }
    return var_map_.at(v);
  }

  Type VisitType_(const TensorTypeNode* op) final {
    GlobalTypeVar gradCell = module_->GetGlobalTypeVar("GradCell");
    tvm::Array<Type> args;
    args.push_back(GetRef<TensorType>(op));
    return TypeCall(gradCell, args);
  }

  Pattern VisitPattern(const Pattern& c) final { return PatternMutator::VisitPattern(c); }

  Constructor VisitConstructor(const Constructor& c) final {
    adt_transformer_->VisitType(c->belong_to);
    return module_->GetConstructor(GradCell_Header + c->belong_to->name_hint,
                                   GradCell_Header + c->name_hint);
  }

  ~LazyGradientInitializer() {
    // destructors
    delete grad_cell_wrapper_;
    delete grad_cell_unwrapper_;
    delete adt_transformer_;
  }

 private:
  // Module
  IRModule module_;

  // pass single instance of ADTTransform to save state of ADTs transformed
  ADTTransform* adt_transformer_;
  // pass single instance of ADTTransform to save state of ADTs wrapped
  GradCellWrapper* grad_cell_wrapper_;
  // pass single instance of ADTTransform to save state of ADTs unwrapped
  GradCellUnWrapper* grad_cell_unwrapper_;
  // var map used for transforming a Clause
  std::unordered_map<Var, Var, ObjectHash, ObjectEqual> var_map_;
  // handle of function -> handle of transformed function
  std::unordered_map<GlobalVar, GlobalVar, ObjectHash, ObjectEqual> func_map_;
  /*!
   * \brief Convert call_node to add/multiply op to use overloaded functions for GradCell type
   */
  Expr CallGradCellFunction(const CallNode* call_node, GlobalVar overloaded_op) {
    // can only use overloaded functions if 2 arguments of same type
    if (call_node->args.size() != 2 ||
        !tvm::StructuralEqual()(call_node->args[0]->checked_type(),
                                call_node->args[1]->checked_type())) {
      Expr result = CallPrimitiveOp(call_node);
      return Call(module_->GetConstructor("GradCell", "Raw"), {result}, Attrs(),
                  {call_node->checked_type()});
    }

    tvm::Array<Expr> args;
    // create "fallback" function for overloaded function
    Type paramType = call_node->args[0]->checked_type();
    tvm::Array<Var> params = {Var("lhs", paramType), Var("rhs", paramType)};
    // use primitive op in this case
    Expr callOp = Call(call_node->op, {params[0], params[1]});
    Expr func = Function(params, callOp, paramType, Array<TypeVar>());

    // pass "fallback" function and tensors as arguments
    args.push_back(func);
    for (Expr expr : call_node->args) {
      args.push_back(VisitExpr(expr));
    }
    // return new call to overloaded function
    return Call(overloaded_op, args, Attrs(), {paramType});
  }

  /*!
   * \brief Convert calls to other ops by converting args into TensorType
   * \return call expr returning result of op
   */
  Expr CallPrimitiveOp(const CallNode* call_node) {
    const auto fromFunc = module_->GetGlobalVar("FromGradCell");
    tvm::Array<Expr> args;

    // unwrap arguments
    for (Expr expr : call_node->args) {
      args.push_back(
          grad_cell_unwrapper_->VisitExpr(VisitExpr(expr), VisitType(expr->checked_type())));
    }
    // result of operation
    return Call(call_node->op, args, call_node->attrs);
  }
};

IRModule LazyGradientInit(const IRModule& m) {
  LazyGradientInitializer lgi = LazyGradientInitializer(m);
  std::vector<GlobalVar> gvs;
  for (const auto& p : m->functions) {
    gvs.push_back(p.first);
  }
  for (const auto& gv : gvs) {
    m->AddUnchecked(gv, lgi.VisitGlobalVar(gv));
  }
  m->Check();
  return m;
}

namespace transform {
Pass LazyGradientInit() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::LazyGradientInit(m); };
  return CreateModulePass(pass_func, 1, "LazyGradientInit", {});
}

TVM_REGISTER_GLOBAL("relay._transform.LazyGradientInit").set_body_typed(LazyGradientInit);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
