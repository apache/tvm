/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_infer.cc
 * \brief Relay type inference and checking.
 *
 * This file implements one of the most important passes to the
 * Relay IR. In order to do many transformations and generate the
 * most efficient code we need to obtain type information for the
 * IR.
 *
 * Like computation graphs the IR leaves most type information
 * implicit and relies performing analysis of the program to
 * generate this information.
 *
 * This pass given an expression `e` will infer a type `t` for
 * the expression simultaneous checking the property `e : t`
 * (i.e we can show e has type t).
 *
 * If we can not infer a type or there are conflicting typing
 * constraints we will trigger an error.
 */

#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/pass/type_infer.h>
#include "./incomplete_type.h"
#include "./resolve.h"
#include "./type_subst.h"
#include "./type_visitor.h"
#include "./unifier.h"
// #include "tvm/relay/typeck/kindchecker.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

struct TypeContext {
  std::vector<std::unordered_map<LocalVar, Type, NodeHash>> stack;

  TypeContext() { stack.push_back({}); }

  void insert(const LocalVar &id, const Type &t) { stack.back()[id] = t; }

  Type lookup(const LocalVar &id) {
    for (auto frame = stack.rbegin(); frame != stack.rend(); ++frame) {
      if (frame->find(id) != frame->end()) {
        return frame->at(id);
      }
    }
    throw FatalTypeError("Could not resolve local id");
  }

  struct LocalFrame {
    TypeContext &tc;
    explicit LocalFrame(TypeContext &tc) : tc(tc) { tc.stack.push_back({}); }
    ~LocalFrame() { tc.stack.pop_back(); }
  };
};

struct TypeNormalizer : TypeFVisitor {
  TypeUnifier unifier;
  TypeNormalizer(const TypeUnifier &unifier) : unifier(unifier) {}

  Type VisitType_(const TypeCallNode *ty_call_node) {
    auto ty_call = GetRef<TypeCall>(ty_call_node);

    Array<Type> normalized_args;

    for (auto arg : ty_call->args) {
      normalized_args.push_back(VisitType(arg));
    }

    auto all_concrete = true;
    for (auto arg : normalized_args) {
      all_concrete = all_concrete && !arg.as<IncompleteTypeNode>();
    }

    if (all_concrete) {
      return normalized_args[normalized_args.size() - 1];
    } else {
      if (auto ty_rel_node = ty_call->func.as<TypeRelationNode>()) {
        // NB: we substract 1 for the output argument.
        auto new_args =
            ty_rel_node->func_(ty_call->args, ty_call->args.size() - 1);
        CHECK(new_args.size() == normalized_args.size());
        tvm::Array<Type> final_args;

        for (int i = 0; i < new_args.size(); i++) {
          final_args.push_back(unifier->unify(normalized_args[i], new_args[i]));
        }

        return TypeCallNode::make(ty_call->func, final_args);
      } else {
        CHECK(false);
      }
    }
  }
};

struct CheckedExpr {
  Expr expr;
  Type type;
  CheckedExpr(Expr e, Type t) : expr(e), type(t) {}
  CheckedExpr() {}
};

class TypeInferencer : private ExprFunctor<CheckedExpr(const Expr &n)> {
 private:
  TypeContext local_stack;

 public:
  Environment env;
  TypeUnifier unifier;

  // Should be in header?
  template <typename T>
  T with_frame(const std::function<T()> &f) {
    TypeContext::LocalFrame fr(local_stack);
    return f();
  }

  TypeInferencer();
  TypeInferencer(Environment env, TypeUnifier unifier)
      : env(env), unifier(unifier) {}
  explicit TypeInferencer(Environment env);

  CheckedExpr Infer(const Expr &expr);

  FuncType instantiate(FuncType fn_ty, tvm::Array<Type> &ty_args);

  Type Normalize(const Type &t);

  void report_error(const std::string &msg, Span sp);
  [[noreturn]] void fatal_error(const std::string &msg, Span sp);

  Type unify(const Type &t1, const Type &t2, Span sp);
  Type resolve(const Type &t);
  Expr resolve(const Expr &e);
  CheckedExpr VisitFunction(const Function &f, bool generalize);
  void CheckOp(Op op);
  // Defn CheckDefn(Defn def);
 private:
  CheckedExpr VisitExpr_(const LocalVarNode *op) override;
  CheckedExpr VisitExpr_(const GlobalVarNode *op) override;
  CheckedExpr VisitExpr_(const ConstantNode *op) override;
  CheckedExpr VisitExpr_(const TupleNode *op) override;
  CheckedExpr VisitExpr_(const ParamNode *op) override;
  CheckedExpr VisitExpr_(const FunctionNode *op) override;
  CheckedExpr VisitExpr_(const CallNode *op) override;
  CheckedExpr VisitExpr_(const LetNode *op) override;
  CheckedExpr VisitExpr_(const IfNode *op) override;
  CheckedExpr VisitExpr_(const OpNode *op) override;
};

TypeInferencer::TypeInferencer() {
  this->env = EnvironmentNode::make({});
  this->unifier = TypeUnifierNode::make(UnionFindNode::make({}));
}

TypeInferencer::TypeInferencer(Environment env) : env(env) {
  this->unifier = TypeUnifierNode::make(UnionFindNode::make({}));
}

Type TypeInferencer::Normalize(const Type &t) {
  auto nt = this->resolve(t);
  auto normalizer = TypeNormalizer(this->unifier);
  return normalizer.VisitType(nt);
}

CheckedExpr TypeInferencer::Infer(const Expr &expr) {
  RELAY_LOG(INFO) << "TypeInferencer::Check expr=" << expr << std::endl;
  CheckedExpr checked_expr = this->VisitExpr(expr);
  RELAY_LOG(INFO) << "TypeInferencer::Check type=" << checked_expr.type
                  << std::endl;
  Type final_type = Normalize(checked_expr.type);
  RELAY_LOG(INFO) << "TypeInferencer::Check type_after_subst=" << final_type
                  << std::endl;
  checked_expr.expr->checked_type_ = final_type;
  return checked_expr;
}

CheckedExpr TypeInferencer::VisitExpr_(const LocalVarNode *op) {
  auto var = GetRef<LocalVar>(op);
  return {var, this->local_stack.lookup(var)};
}

CheckedExpr TypeInferencer::VisitExpr_(const GlobalVarNode *op) {
  GlobalVar var = GetRef<GlobalVar>(op);
  Expr e = this->env->Lookup(var);
  return { var, e->checked_type() };
}

CheckedExpr TypeInferencer::VisitExpr_(const ConstantNode *const_node) {
  return {GetRef<Constant>(const_node), const_node->tensor_type()};
}

CheckedExpr TypeInferencer::VisitExpr_(const TupleNode *op) {
  Tuple pl = GetRef<Tuple>(op);

  std::vector<Expr> field_exprs;
  std::vector<Type> field_types;
  for (auto field = pl->fields.begin(); field != pl->fields.end(); field++) {
    auto checked_field = Infer(*field);
    field_exprs.push_back(checked_field.expr);
    field_types.push_back(checked_field.type);
  }

  return { TupleNode::make(field_exprs), TupleTypeNode::make(field_types) };
}

CheckedExpr TypeInferencer::VisitExpr_(const ParamNode *param) {
  auto rtype = resolve(param->type);
  return {ParamNode::make(param->var, rtype), rtype};
}

// // We should probably generalize the subst code.
// struct GeneralizeTypeType : TypeFVisitor {
//   Map<TypeVar, TypeParam> vars_to_id;
//   const TypeUnifier &unifier;

//   GeneralizeTypeType(Map<TypeVar, TypeParam> vars_to_id,
//                      const TypeUnifier &unifier)
//       : vars_to_id(vars_to_id), unifier(unifier) {}

//   Type VisitType_(const TypeVarNode *op) override {
//     auto repr = unifier->subst(GetRef<TypeVar>(op));
//     if (auto tvn = repr.as<TypeVarNode>()) {
//       auto ty_var = GetRef<TypeVar>(tvn);
//       if (vars_to_id.find(ty_var) != vars_to_id.end()) {
//         return vars_to_id[ty_var];
//       } else {
//         return ty_var;
//       }
//     } else {
//       return this->VisitType(repr);
//     }
//   }
// };

// struct GeneralizeTypeExpr : ExprFVisitor<> {
//   Map<TypeVar, TypeParam> vars_to_id;
//   const TypeUnifier &unifier;

//   GeneralizeTypeExpr(const TypeUnifier &unifier,
//                      Map<TypeVar, TypeParam> vars_to_id)
//       : vars_to_id(vars_to_id), unifier(unifier) {}

//   Type VisitType(const Type &t) {
//     return GeneralizeTypeType(vars_to_id, unifier).VisitType(t);
//   }
// };

CheckedExpr TypeInferencer::VisitFunction(const Function &f, bool generalize) {
  // First we add the parameters to the context allowing us to check their
  // types.

  // TODO(@jroesch): support polymorphism

  std::vector<Type> param_types;
  std::vector<Param> params;

  return this->with_frame<CheckedExpr>([&]() -> CheckedExpr {
    for (auto param : f->params) {
      CheckedExpr checked_param = this->Infer(param);
      Type arg_type;
      param_types.push_back(checked_param.type);
      params.push_back(GetRef<Param>(checked_param.expr.as<ParamNode>()));
      this->local_stack.insert(param->var, checked_param.type);
    }

    auto checked_body = this->Infer(f->body);
    auto inferred_rtype = checked_body.type;
    auto annotated_rtype = resolve(f->ret_type);

    auto unified_rtype = this->unify(inferred_rtype, annotated_rtype, f->span);

    return {FunctionNode::make(params, unified_rtype, checked_body.expr, {}),
            FuncTypeNode::make(param_types, unified_rtype, {}, {})};
  });

  //   // typecheck body and ensure that it matches stated return type
  //   // TODO(sslyu): should the unified return type override the annotated
  //   one? Type checked_return = this->Check(f->body); Type ret_type =
  //   resolve(f->ret_type); Type unified =
  //   this->unify(simple_eval_shape(ret_type),
  //                              simple_eval_shape(checked_return), f->span);
  //   return TypeArrowNode::make(arg_types, unified);
  // });
  // if (generalize) {
  //   auto free_vars = free_type_vars(resolve(fn_type));
  //   std::set<TypeVar> dedup_free_vars;

  //   for (auto free_var : free_vars) {
  //     auto repr = this->unifier->subst(free_var);
  //     if (auto new_free_var_node = repr.as<TypeVarNode>()) {
  //       dedup_free_vars.insert(GetRef<TypeVar>(new_free_var_node));
  //     } else {
  //       // debug(repr);
  //       throw dmlc::Error(
  //           "internal error: this list should only contain type var
  //           nodes");
  //     }
  //   }

  //   Map<TypeVar, TypeParam> vars_to_id;

  //   GenFresh gf;
  //   for (auto free_var : dedup_free_vars) {
  //     vars_to_id.Set(free_var, gf.freshTV(free_var->kind));
  //   }

  //   fn_type = GeneralizeTypeType(vars_to_id, unifier).VisitType(fn_type);
  //   for (std::pair<TypeVar, TypeParam> pair : vars_to_id) {
  //     // NB: In generalization we want to find type variables with
  //     // *no constraints* on them, and convert them to universally
  //     quantified
  //     // variables.
  //     //
  //     // i.e the program can be abstracted over the details of *that* type.

  //     // For example a program that works irrespective of shape or
  //     datatype.

  //     // In order to do this we find the set of free type variables in the
  //     // term, and then unify them with the fresh type ids we generate.
  //     //
  //     // Remember importantly these type variables still may appear in many
  //     // places in the program including both types and expressions.

  //     // Our method for resolving these is to unify them with the variables
  //     // as we build the new quanitifer, changing from a program with
  //     "holes"
  //     // to one that is properly abstracted over.

  //     // Finally later on we can iterate over the whole term and change
  //     from
  //     // type variables to these type ids.
  //     this->unify(pair.first, pair.second, pair.second->span);
  //     fn_type = TypeQuantifierNode::make(pair.second, fn_type);
  //   }
  // } else {
  //   for (auto i = f->ty_params.size(); i > 0; i--) {
  //     auto ty_param = f->ty_params[i - 1];
  //     auto ty_param_node = ty_param.as<TypeParamNode>();
  //     if (!ty_param_node) {
  //       throw dmlc::Error("internal error should be TypeParam");
  //     }
  //     auto fresh_tid =
  //         TypeParamNode::make(ty_param_node->name, ty_param_node->kind);
  //     fn_type =
  //         TypeSubst(fn_type, GetRef<TypeParam>(ty_param_node), fresh_tid);
  //     fn_type = TypeQuantifierNode::make(fresh_tid, fn_type);
  //   }
  // }

  // return fn_type;
}

CheckedExpr TypeInferencer::VisitExpr_(const FunctionNode *op) {
  return this->VisitFunction(GetRef<Function>(op), false);
}

FuncType TypeInferencer::instantiate(FuncType fn_ty,
                                     tvm::Array<Type> &ty_args) {
  tvm::Map<TypeParam, Type> subst_map;

  // Build a subsitituion map up from the function type and type arguments.
  // Eventually allow the type vars to be passed in.
  for (auto ty_param : fn_ty->type_params) {
    IncompleteType fresh = IncompleteTypeNode::make(ty_param->kind);
    this->unifier->insert(fresh);
    ty_args.push_back(fresh);
    subst_map.Set(ty_param, fresh);
  }

  Type inst_ty = FuncTypeNode::make(fn_ty->arg_types, fn_ty->ret_type, {}, {});
  inst_ty = TypeSubst(fn_ty, subst_map);

  // if (!check_kind(t)) {
  //   this->fatal_error("Kind rules broken when instantiating type
  //   variables",
  //                     t->span);
  // }

  return GetRef<FuncType>(inst_ty.as<FuncTypeNode>());
}

CheckedExpr TypeInferencer::VisitExpr_(const CallNode *op) {
  Call c = GetRef<Call>(op);

  auto checked_op = this->Infer(c->op);

  RELAY_LOG(INFO) << "TypeInferencer::VisitExpr_ op=" << c << std::endl
                  << "fn_ty=" << checked_op.type << std::endl;

  auto fn_ty_node = checked_op.type.as<FuncTypeNode>();

  if (!fn_ty_node) {
    this->fatal_error("only expressions with function types can be called",
                      c->op->span);
  }

  // We now have a function type.
  FuncType fn_ty = GetRef<FuncType>(fn_ty_node);

  tvm::Array<Type> ty_args;
  if (ty_args.size() != 0) {
    throw Error("found manually suplied type args, not supported");
  }

  fn_ty = instantiate(fn_ty, ty_args);

  std::vector<Type> arg_types;
  std::vector<Expr> checked_args;

  for (auto arg : c->args) {
    auto checked_arg = this->Infer(arg);
    arg_types.push_back(checked_arg.type);
    checked_args.push_back(checked_arg.expr);
  }

  auto type_arity = fn_ty->arg_types.size();
  auto number_of_args = arg_types.size();

  if (type_arity != number_of_args) {
    if (type_arity < number_of_args) {
      this->fatal_error("the function is provided too many arguments", c->span);
    } else {
      this->fatal_error("the function is provided too few arguments", c->span);
    }
  }

  for (size_t i = 0; i < fn_ty->arg_types.size(); i++) {
    this->unify(fn_ty->arg_types[i], arg_types[i], c->args[i]->span);
  }

  // After we unify the arguments we should know more about the type
  // arguments, let's run a quick pass over them to find new
  // representatives.

  for (size_t i = 0; i < ty_args.size(); i++) {
    ty_args.Set(i, this->unifier->subst(ty_args[i]));
  }

  auto new_call =
      CallNode::make(checked_op.expr, checked_args, c->attrs, ty_args);

  return {new_call, fn_ty->ret_type};
}

CheckedExpr TypeInferencer::VisitExpr_(const LetNode *op) {
  Let let = GetRef<Let>(op);

  CheckedExpr checked_value;
  Type annotated_ty = resolve(let->value_type);

  // If we are let-defining a function, we want to be able to
  // recursively name the function in order to support recursive
  // local definitions.
  if (let->value.as<FunctionNode>()) {
    with_frame<void>([&]() {
      local_stack.insert(let->var, annotated_ty);
      checked_value = Infer(let->value);
    });
  } else {
    checked_value = Infer(let->value);
  }

  Type unified_ty = this->unify(checked_value.type, annotated_ty, let->span);

  // Update type context with unified type now that we have
  // solved this equation.
  local_stack.insert(let->var, unified_ty);

  auto checked_body = with_frame<CheckedExpr>([&]() {
    local_stack.insert(let->var, unified_ty);
    return Infer(let->body);
  });

  auto checked_let = LetNode::make(let->var, checked_value.expr,
                                   checked_body.expr, let->value_type);

  return {checked_let, checked_body.type};
}

CheckedExpr TypeInferencer::VisitExpr_(const IfNode *op) {
  If ifn = GetRef<If>(op);

  // Ensure the type of the guard is of Tensor[Bool, ()],
  // that is a rank-0 boolean tensor.
  auto checked_cond = this->Infer(ifn->cond);
  auto cond_type = checked_cond.type;

  if (const TensorTypeNode *tt_node = cond_type.as<TensorTypeNode>()) {
    TensorType tt = GetRef<TensorType>(tt_node);
    if (tt->dtype.is_bool() && tt->shape.size() == 0) {
      auto checked_true = this->Infer(ifn->true_value);
      auto checked_false = this->Infer(ifn->false_value);
      auto unified_type =
          this->unify(checked_true.type, checked_false.type, ifn->span);
      auto checked_if = IfNode::make(checked_cond.expr, checked_true.expr,
                                     checked_false.expr);
      return {checked_if, unified_type};
    }
  }

  this->fatal_error("if-then-else guard must be a rank-0 boolean tensor",
                    ifn->cond->span);
}

CheckedExpr TypeInferencer::VisitExpr_(const OpNode *op_node) {
  auto op = GetRef<Op>(op_node);
  return {op, op->op_type};
}

Type TypeInferencer::resolve(const Type &t) {
  if (t.defined()) {
    return ::tvm::relay::Resolve(this->unifier, t);
  } else {
    return IncompleteTypeNode::make(TypeParamNode::Kind::kType);
  }
}

Expr TypeInferencer::resolve(const Expr &e) {
  CHECK(e.defined());
  return ::tvm::relay::Resolve(this->unifier, e);
}

void TypeInferencer::CheckOp(Op op) {
  throw Error("NYI");
  // if (!check_kind(op->type)) {
  //   report_error("the type of the operator is ill formed", op->type->span);
  // }

  // // Fix me
  // return op;
}

// Defn TypeInferencer::CheckDefn(Defn defn) {
//   // This is to handle recursion, but we need to speculatively
//   // put it in env, then remove it.
//   env->items.insert({defn->id, defn});

//   Type expected_ty = this->resolve(defn->type);

//   Expr body = defn->body;

//   auto checked_ty = Check(body);

//   try {
//     Type uret_type = unify(expected_ty, checked_ty, defn->body->span);
//     CHECK(is_fully_resolved(uret_type));
//     // Now let's clean up our work from earlier.
//     env->items.erase(defn->id);
//     return DefnNode::make(defn->id, uret_type, this->resolve(defn->body));
//   } catch (const UnificationError& err) {
//       std::string msg = std::string("mismatch between `") +
//                         PrintType(env, expected_ty, WrapWidth(40)) + "` and
//                         `" + PrintType(env, checked_ty, WrapWidth(40)) +
//                         "`";
//       fatal_error(msg, defn->span);
//   }
// }

Expr InferType(const Environment &env, const Expr &e) {
  TypeInferencer ti(env);
  auto checked_expr = ti.Infer(e);
  return checked_expr.expr;
}

// Item Check(const Environment &env, const Item &i) {
//   TypeInferencer tc(env);

//   try {
//     if (const DefnNode *defn = i.as<DefnNode>()) {
//       return tc.CheckDefn(GetRef<Defn>(defn));
//     } else if (const OpNode *op_node = i.as<OpNode>()) {
//       return tc.CheckOp(GetRef<Op>(op_node));
//     } else {
//       throw dmlc::Error("internal error: unknown Item type");
//     }
//   } catch (const FatalTypeError &err) {
//     env->display_errors();
//     throw dmlc::Error(
//         "We encountered a fatal error while type checking your program,
//         please " "read above for more details.");
//   }
// }

inline void TypeInferencer::report_error(const std::string &msg, Span sp) {
  // this->env->report_error(msg, sp);
}

void TypeInferencer::fatal_error(const std::string &msg, Span sp) {
  // this->env->report_error(msg, sp);
  throw FatalTypeError(
      "internal error: this exception should"
      "be handled and errors reported with Environment::display_errors\n" +
      msg);
}

Type TypeInferencer::unify(const Type &t1, const Type &t2, Span sp) {
  try {
    return Normalize(this->unifier->unify(t1, t2));
  } catch (const dmlc::Error &e) {
    std::stringstream ss;
    ss << "Error unifying `";
    ss << t1;
    // ss << PrintType(env, t1, WrapWidth(40));
    ss << "` and `";
    ss << t2;
    // ss << PrintType(env, t2, WrapWidth(40));
    ss << "`: " << e.what();
    this->fatal_error(ss.str(), sp);
  }
}

TVM_REGISTER_API("relay._ir_pass.check_expr")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      Expr e = args[1];
      *ret = InferType(env, e);
    });

// TODO(@jroesch): put in a better namespace.
TVM_REGISTER_API("relay._ir_pass._get_checked_type")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Expr e = args[0];
      *ret = e->checked_type();
    });

IncompleteType IncompleteTypeNode::make(TypeParamNode::Kind kind) {
  std::shared_ptr<IncompleteTypeNode> n =
      std::make_shared<IncompleteTypeNode>();
  n->kind = std::move(kind);
  return IncompleteType(n);
}

TVM_REGISTER_API("relay._make.IncompleteType")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      int kind = args[0];
      *ret = IncompleteTypeNode::make(static_cast<TypeParamNode::Kind>(kind));
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<IncompleteTypeNode>([](const IncompleteTypeNode *node,
                                         tvm::IRPrinter *p) {
      p->stream << "IncompleteTypeNode(" << node->kind << ", " << node << ")";
    });

}  // namespace relay
}  // namespace tvm
