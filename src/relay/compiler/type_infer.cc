/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_infer.cc
 * \brief Relay type inference and checking.
 */

#include "tvm/relay/compiler/type_infer.h"
#include "./incomplete_type.h"
// #include "tvm/relay/alpha_eq.h"
// #include "tvm/relay/debug.h"
// #include "tvm/relay/first_order_reverse_ad.h"
// #include "tvm/relay/free_type_vars.h"
// #include "tvm/relay/gen_fresh.h"
// #include "tvm/relay/ir.h"
// #include "tvm/relay/logging.h"
// #include "tvm/relay/pretty_printer.h"
// #include "tvm/relay/reverse_ad.h"
// #include "tvm/relay/type_visitor.h"
// #include "tvm/relay/typeck/kindchecker.h"
// #include "tvm/relay/typeck/resolve.h"
// #include "tvm/relay/typeck/shape_evaluator.h"

namespace tvm {
namespace relay {

// using namespace tvm::runtime;

// struct FatalTypeError : dmlc::Error {
//   explicit FatalTypeError(const std::string & s) : dmlc::Error(s) {}
// };

// struct TypeContext {
//   std::vector<std::unordered_map<LocalId, Type>> stack;
//   TypeContext() {
//     stack.push_back({});
//   }
//   void insert(const LocalId &id, const Type &t) { stack.back()[id] = t; }
//   Type lookup(const LocalId &id) {
//     for (auto frame = stack.rbegin(); frame != stack.rend(); ++frame) {
//       if (frame->find(id) != frame->end()) {
//         return frame->at(id);
//       }
//     }
//     throw FatalTypeError("Could not resolve local id");
//   }
//   struct LocalFrame {
//     TypeContext & tc;
//     explicit LocalFrame(TypeContext & tc) : tc(tc) {
//       tc.stack.push_back({});
//     }
//     ~LocalFrame() {
//       tc.stack.pop_back();
//     }
//   };
// };

// class Typechecker : private ExprFunctor<Type(const Expr & n)> {
//  private:
//   TypeContext local_stack;
//  public:
//   Environment env;
//   TypeUnifier unifier;

//   template<typename T>
//     T with_frame(const std::function<T()> & f) {
//     TypeContext::LocalFrame fr(local_stack);
//     return f();
//   }

//   Typechecker();
//   Typechecker(Environment env, TypeUnifier unifier) : env(env), unifier(unifier) {}
//   explicit Typechecker(Environment env);
//   Type Check(const Expr & expr);
//   Type instantiate(Type t, tvm::Array<Type> & ty_args);

//   void report_error(const std::string & msg, Span sp);
//   [[ noreturn ]] void fatal_error(const std::string & msg, Span sp);

//   Type unify(const Type &t1, const Type &t2, Span sp);
//   Type resolve(const Type &t);
//   Expr resolve(const Expr &e);
//   Type VisitFunction(const Function & f, bool generalize);
//   Operator CheckOp(Operator op);
//   Defn CheckDefn(Defn def);
//  private:
//   Type VisitExpr_(const LocalIdNode* op) override;
//   Type VisitExpr_(const GlobalIdNode* op) override;
//   Type VisitExpr_(const OperatorIdNode* op) override;
//   Type VisitExpr_(const FloatLitNode* op) override;
//   Type VisitExpr_(const BoolLitNode* op) override;
//   Type VisitExpr_(const IntLitNode* op) override;
//   Type VisitExpr_(const TensorLitNode* op) override;
//   Type VisitExpr_(const TupleNode* op) override;
//   Type VisitExpr_(const CastNode* op) override;
//   Type VisitExpr_(const ParamNode* op) override;
//   Type VisitExpr_(const FunctionNode* op) override;
//   Type VisitExpr_(const CallNode* op) override;
//   Type VisitExpr_(const DebugNode* op) override;
//   Type VisitExpr_(const LetNode* op) override;
//   Type VisitExpr_(const ReverseNode* op) override;
//   Type VisitExpr_(const GradientNode* op) override;
//   Type VisitExpr_(const ProjectionNode* op) override;
//   Type VisitExpr_(const IfNode* op) override;
//   Type VisitExpr_(const RefNode* op) override;
//   Type VisitExpr_(const ReadRefNode* op) override;
//   Type VisitExpr_(const WriteRefNode* op) override;
//   Type simple_eval_shape(const Type &shape);
// };
// struct TypecheckerError : public dmlc::Error {
//   explicit TypecheckerError(const std::string &msg) : Error(msg) {}
// };

// Typechecker::Typechecker() {
//   this->env = EnvironmentNode::make({});
//   this->unifier = TypeUnifierNode::make(UnionFindNode::make({}));
// }

// Typechecker::Typechecker(Environment env) : env(env) {
//   this->unifier = TypeUnifierNode::make(UnionFindNode::make({}));
// }

// Type Typechecker::Check(const Expr &expr) {
//   RELAY_LOG(INFO) << "Typechecker::Check expr=" << expr << std::endl;
//   Type ret = this->VisitExpr(expr);
//   RELAY_LOG(INFO) << "Typechecker::Check type=" << expr << std::endl;
//   ret = this->unifier->subst(ret);
//   RELAY_LOG(INFO) << "Typechecker::Check type_after_subst=" << ret << std::endl;
//   expr->checked_type_ = ret;
//   return ret;
// }

// Type Typechecker::VisitExpr_(const LocalIdNode *op) {
//   LocalId id = GetRef<LocalId>(op);
//   return this->local_stack.lookup(id);
// }

// Type Typechecker::VisitExpr_(const GlobalIdNode *op) {
//   GlobalId id = GetRef<GlobalId>(op);
//   Item item = this->env->lookup(id);

//   if (const OperatorNode *op = item.as<OperatorNode>()) {
//     return op->type;
//   }

//   if (const DefnNode *dn = item.as<DefnNode>()) {
//     Defn def = GetRef<Defn>(dn);
//     return def->type;
//   }

//   this->fatal_error("Unhandled case in GlobalId", op->span);
// }

// Type Typechecker::VisitExpr_(const OperatorIdNode *op) {
//   OperatorId id = GetRef<OperatorId>(op);
//   Item item = this->env->lookup(id);

//   if (const OperatorNode *pn = item.as<OperatorNode>()) {
//     Operator prim = GetRef<Operator>(pn);
//     return prim->type;
//   } else {
//     this->fatal_error("internal error in InstrinsicId case", op->span);
//   }
// }

// Type Typechecker::VisitExpr_(const FloatLitNode *op) { return FloatType(); }

// Type Typechecker::VisitExpr_(const BoolLitNode *op) { return BoolType(); }

// Type Typechecker::VisitExpr_(const IntLitNode *op) { return IntType(); }

// Type Typechecker::VisitExpr_(const TensorLitNode *op) {
//   TensorLit lit = GetRef<TensorLit>(op);

//   if (lit->data.size() == 0) {
//     this->fatal_error("Tensor literal must have at least one member", op->span);
//   }

//   // unify types of all members to figure out shape, also ensure that
//   // each member has compatible shape
//   Type unified = this->Check(lit->data[0]);
//   for (auto elt = lit->data.begin(); elt != lit->data.end(); elt++) {
//     // evaluate all shape ASTs so they can be in standard form
//     // TODO(sslyu): eventually we'd want this to be symbolic evaluation
//     auto elt_el = *elt;
//     Type elt_type = simple_eval_shape(this->Check(*elt));
//     if (!elt_type.as<TensorTypeNode>()) {
//       this->fatal_error("All members in tensor literal must be tensors",
//                         elt_el->span);
//     }
//     unified = this->unify(unified, elt_type, lit->span);
//   }

//   // types must unify into a tensor
//   const TensorTypeNode *ttn = unified.as<TensorTypeNode>();
//   // shouldn't be possible due to check inside the loop
//   if (!ttn) {
//     this->fatal_error("Tensor literal contains non-tensor member", op->span);
//   }

//   TensorType unified_tt = GetRef<TensorType>(ttn);

//   // new shape: add length of this tensor to front of existing shape
//   // i.e., sequence and simplify
//   // TODO(sslyu): should be symbolic evaluation eventually?
//   Type new_shape = ShapeSeqNode::make(
//       {ShapeSingletonNode::make(lit->data.size()), unified_tt->shape});
//   return TensorTypeNode::make(unified_tt->dtype, simple_eval_shape(new_shape));
// }

// Type Typechecker::VisitExpr_(const TupleNode *op) {
//   Tuple pl = GetRef<Tuple>(op);

//   std::vector<Type> field_types;
//   for (auto field = pl->fields.begin(); field != pl->fields.end(); field++) {
//     field_types.push_back(this->Check(*field));
//   }

//   return TupleTypeNode::make(field_types);
// }

// Type Typechecker::VisitExpr_(const CastNode *op) {
//   // will take the cast at its word
//   Cast cast = GetRef<Cast>(op);
//   return cast->target;
// }

// Type Typechecker::VisitExpr_(const ParamNode *op) {
//   Param param = GetRef<Param>(op);
//   return resolve(param->type);
// }

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

// Type Typechecker::VisitFunction(const Function &f, bool generalize) {
//   // enter params into context
//   auto fn_type = this->with_frame<Type>([&]() {
//     std::vector<Type> arg_types;
//     for (auto arg : f->params) {
//       this->Check(arg);
//       Type arg_type;
//       // if arg type can be simply evaluated, try it
//       // should be replaced with symbolic evaluation once it exists,
//       // you will not have attr information at this point
//       try {
//         arg_type = simple_eval_shape(arg->type);
//       } catch (const dmlc::Error &e) {
//         this->report_error(e.what(), arg->span);
//         arg_type = arg->type;
//       }
//       arg_types.push_back(arg_type);
//       this->local_stack.insert(arg->id, arg_type);
//     }

//     // typecheck body and ensure that it matches stated return type
//     // TODO(sslyu): should the unified return type override the annotated one?
//     Type checked_return = this->Check(f->body);
//     Type ret_type = resolve(f->ret_type);
//     Type unified = this->unify(simple_eval_shape(ret_type),
//                                simple_eval_shape(checked_return), f->span);
//     return TypeArrowNode::make(arg_types, unified);
//   });
//   if (generalize) {
//     auto free_vars = free_type_vars(resolve(fn_type));
//     std::set<TypeVar> dedup_free_vars;

//     for (auto free_var : free_vars) {
//       auto repr = this->unifier->subst(free_var);
//       if (auto new_free_var_node = repr.as<TypeVarNode>()) {
//         dedup_free_vars.insert(GetRef<TypeVar>(new_free_var_node));
//       } else {
//         // debug(repr);
//         throw dmlc::Error(
//             "internal error: this list should only contain type var nodes");
//       }
//     }

//     Map<TypeVar, TypeParam> vars_to_id;

//     GenFresh gf;
//     for (auto free_var : dedup_free_vars) {
//       vars_to_id.Set(free_var, gf.freshTV(free_var->kind));
//     }

//     fn_type = GeneralizeTypeType(vars_to_id, unifier).VisitType(fn_type);
//     for (std::pair<TypeVar, TypeParam> pair : vars_to_id) {
//       // NB: In generalization we want to find type variables with
//       // *no constraints* on them, and convert them to universally quantified
//       // variables.
//       //
//       // i.e the program can be abstracted over the details of *that* type.

//       // For example a program that works irrespective of shape or datatype.

//       // In order to do this we find the set of free type variables in the
//       // term, and then unify them with the fresh type ids we generate.
//       //
//       // Remember importantly these type variables still may appear in many
//       // places in the program including both types and expressions.

//       // Our method for resolving these is to unify them with the variables
//       // as we build the new quanitifer, changing from a program with "holes"
//       // to one that is properly abstracted over.

//       // Finally later on we can iterate over the whole term and change from
//       // type variables to these type ids.
//       this->unify(pair.first, pair.second, pair.second->span);
//       fn_type = TypeQuantifierNode::make(pair.second, fn_type);
//     }
//   } else {
//     for (auto i = f->ty_params.size(); i > 0; i--) {
//       auto ty_param = f->ty_params[i - 1];
//       auto ty_param_node = ty_param.as<TypeParamNode>();
//       if (!ty_param_node) {
//         throw dmlc::Error("internal error should be TypeParam");
//       }
//       auto fresh_tid =
//           TypeParamNode::make(ty_param_node->name, ty_param_node->kind);
//       fn_type =
//           type_subst(fn_type, GetRef<TypeParam>(ty_param_node), fresh_tid);
//       fn_type = TypeQuantifierNode::make(fresh_tid, fn_type);
//     }
//   }

//   return fn_type;
// }

// Type Typechecker::VisitExpr_(const FunctionNode *op) {
//   return this->VisitFunction(GetRef<Function>(op), false);
// }

// Type Typechecker::instantiate(Type t, tvm::Array<Type> &ty_args) {
//   const TypeQuantifierNode *ty_quant;
//   while ((ty_quant = t.as<TypeQuantifierNode>())) {
//     TypeParam id = ty_quant->id;
//     TypeVar fresh = TypeVarNode::make(id->kind);
//     this->unifier->insert(fresh);
//     ty_args.push_back(fresh);
//     t = type_subst(ty_quant->boundType, id, fresh);
//   }

//   if (!check_kind(t)) {
//     this->fatal_error("Kind rules broken when instantiating type variables",
//                       t->span);
//   }

//   return t;
// }

// Type Typechecker::VisitExpr_(const CallNode *op) {
//   Call c = GetRef<Call>(op);
//   Type fn_ty = this->Check(c->fn);

//   RELAY_LOG(INFO) << "Typechecker::VisitExpr_ op=" << c << std::endl
//                   << "fn_ty=" << fn_ty << std::endl;

//   // for each type id, insert a type variable and unify with the argument types
//   // in order
//   // to obtain the concrete instantiation
//   tvm::Array<Type> ty_args;
//   if (const TypeQuantifierNode *ty_quant = fn_ty.as<TypeQuantifierNode>()) {
//     fn_ty = instantiate(GetRef<TypeQuantifier>(ty_quant), ty_args);
//   }

//   if (!fn_ty.as<TypeArrowNode>()) {
//     this->fatal_error("only expressions with function types can be called",
//                       c->fn->span);
//   }

//   // evaluate all shapes up front (require that types be fully concrete)
//   Type evaluated = evaluate_concrete_shape(fn_ty, op->attrs);
//   std::vector<Type> arg_types;

//   TypeArrow arrow = GetRef<TypeArrow>(evaluated.as<TypeArrowNode>());

//   // TODO(sslyu): figure out how to handle type ids
//   //  fn_ty = instantiate(fn_ty, ty_args);
//   for (auto arg : c->args) {
//     auto ty = this->Check(arg);
//     arg_types.push_back(ty);
//   }

//   auto type_arity = arrow->arg_types.size();
//   auto number_of_args = arg_types.size();
//   if (type_arity != number_of_args) {
//     if (type_arity < number_of_args) {
//       this->fatal_error("the function is provided too many arguments", c->span);
//     } else {
//       this->fatal_error("the function is provided too few arguments", c->span);
//     }
//   }

//   for (size_t i = 0; i < arrow->arg_types.size(); i++) {
//     this->unify(arrow->arg_types[i], arg_types[i], c->args[i]->span);
//   }

//   // After we unify the arguments we should know more about the type
//   // arguments, let's run a quick pass over them to find new representatives.
//   for (size_t i = 0; i < ty_args.size(); i++) {
//     ty_args.Set(i, this->unifier->subst(ty_args[i]));
//   }

//   // Write the type arguments into the call node, recording what inference
//   // solves. This solution might need some work.
//   c->ty_args = ty_args;

//   return arrow->ret_type;
// }

// Type Typechecker::VisitExpr_(const DebugNode *op) {
//   return this->Check(op->node);
// }

// Type Typechecker::VisitExpr_(const LetNode *op) {
//   Let let = GetRef<Let>(op);

//   Type checked_ty;
//   Type annotated_ty = resolve(let->type);

//   // if we are let-defining a function, treat it as a let-rec and insert
//   // the id with the annotated type in case there is recursion;
//   // no such recursion permitted with anything that's not a function!
//   if (let->value.as<FunctionNode>()) {
//     with_frame<void>([&]() {
//       local_stack.insert(let->id, annotated_ty);
//       checked_ty = Check(let->value);
//     });
//   } else {
//     checked_ty = Check(let->value);
//   }

//   // ensure annotated type and checked type are compatible
//   // TODO(sslyu): should the annotated type override the unified one?
//   Type unified_ty =
//       this->unify(checked_ty, simple_eval_shape(annotated_ty), let->span);

//   return with_frame<Type>([&]() {
//     local_stack.insert(let->id, unified_ty);
//     return Check(let->body);
//   });
// }

// Type Typechecker::VisitExpr_(const ReverseNode *op) {
//   // apply reverse mode to node and typecheck that instead
//   std::shared_ptr<GenFresh> gf = std::make_shared<GenFresh>();
//   return this->Check(ReverseExpr(env, op->node, gf));
// }

// Type Typechecker::VisitExpr_(const GradientNode *op) {
//   auto node = op->node;
//   this->Check(node);
//   auto gf = std::make_shared<GenFresh>();
//   return FOWithGradientType(node->checked_type());
// }

// Type Typechecker::VisitExpr_(const ProjectionNode *op) {
//   Projection proj = GetRef<Projection>(op);

//   Type tup_type = this->Check(proj->tuple);

//   const TupleTypeNode *ptn = tup_type.as<TupleTypeNode>();
//   if (!ptn) {
//     this->fatal_error("Cannot project into non-product type", op->span);
//   }

//   TupleType pt = GetRef<TupleType>(ptn);
//   size_t field = (size_t)proj->field;
//   if (field >= pt->fields.size()) {
//     this->fatal_error("Projecting past bounds of product", op->span);
//   }

//   return pt->fields[field];
// }

// Type Typechecker::VisitExpr_(const IfNode *op) {
//   If ifn = GetRef<If>(op);

//   // Ensure the type of the guard is of Tensor[Bool, ()],
//   // that is a rank-0 boolean tensor.
//   Type guardType = this->Check(ifn->guard);
//   bool is_bool = false;
//   bool zero_rank = false;
//   if (const TensorTypeNode *ttn = guardType.as<TensorTypeNode>()) {
//     TensorType tt = GetRef<TensorType>(ttn);

//     if (const BaseTypeNode *btn = tt->dtype.as<BaseTypeNode>()) {
//       is_bool = btn->type.is_bool();
//     }

//     Type shape = simple_eval_shape(tt->shape);

//     if (const ShapeSeqNode *sn = shape.as<ShapeSeqNode>()) {
//       zero_rank = (sn->shapes.size() == 0);
//     }
//   }

//   if (!(is_bool && zero_rank)) {
//     this->fatal_error("IfNode guard must be a rank 0 bool tensor",
//                       ifn->guard->span);
//   }

//   // unify types of different branches
//   Type left = this->Check(ifn->true_b);
//   Type right = this->Check(ifn->false_b);
//   return this->unify(left, right, ifn->span);
// }

// Type Typechecker::VisitExpr_(const RefNode *op) {
//   Ref r = GetRef<Ref>(op);
//   Type inner = this->Check(r->expr);
//   return RefTypeNode::make(inner);
// }

// Type Typechecker::VisitExpr_(const ReadRefNode *op) {
//   ReadRef vr = GetRef<ReadRef>(op);
//   Type ref_type = this->Check(vr->ref);

//   // reject if not a ref type
//   const RefTypeNode *rtn = ref_type.as<RefTypeNode>();
//   if (!rtn) {
//     this->fatal_error(
//         "the de-reference operation can only be used with references",
//         op->span);
//   }

//   RefType rt = GetRef<RefType>(rtn);
//   return rt->data_type;
// }

// Type Typechecker::VisitExpr_(const WriteRefNode *op) {
//   WriteRef sr = GetRef<WriteRef>(op);
//   Type ref_type = this->Check(sr->ref);

//   const RefTypeNode *rtn = ref_type.as<RefTypeNode>();
//   if (!rtn) {
//     this->fatal_error("Cannot mutate non-ref", op->span);
//   }
//   RefType rt = GetRef<RefType>(rtn);

//   // ensure ref type's inner type and expr's type are compatible; return unit
//   Type expr_type = this->Check(sr->val);
//   this->unify(rt->data_type, expr_type, sr->span);
//   return UnitType();
// }

// Type Typechecker::resolve(const Type &t) {
//   return ::tvm::relay::resolve(this->unifier, t);
// }

// Expr Typechecker::resolve(const Expr &e) {
//   return ::tvm::relay::resolve(this->unifier, e);
// }

// Type Typechecker::simple_eval_shape(const Type &shape) {
//   // TODO(sslyu): Do we want to propagate attributes?
//   Attributes empty = AttributesNode::make({});
//   return evaluate_concrete_shape(shape, empty);
// }

// Operator Typechecker::CheckOp(Operator op) {
//   if (!check_kind(op->type)) {
//     report_error("the type of the operator is ill formed", op->type->span);
//   }

//   // Fix me
//   return op;
// }

// Defn Typechecker::CheckDefn(Defn defn) {
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
//                         PrintType(env, expected_ty, WrapWidth(40)) + "` and `" +
//                         PrintType(env, checked_ty, WrapWidth(40)) + "`";
//       fatal_error(msg, defn->span);
//   }
// }

// Type check(const Environment &env, const Expr &e) {
//   Typechecker tc(env);
//   return tc.Check(e);
// }

// Item check(const Environment &env, const Item &i) {
//   Typechecker tc(env);

//   try {
//     if (const DefnNode *defn = i.as<DefnNode>()) {
//       return tc.CheckDefn(GetRef<Defn>(defn));
//     } else if (const OperatorNode *op_node = i.as<OperatorNode>()) {
//       return tc.CheckOp(GetRef<Operator>(op_node));
//     } else {
//       throw dmlc::Error("internal error: unknown Item type");
//     }
//   } catch (const FatalTypeError &err) {
//     env->display_errors();
//     throw dmlc::Error(
//         "We encountered a fatal error while type checking your program, please "
//         "read above for more details.");
//   }
// }

// inline void Typechecker::report_error(const std::string &msg, Span sp) {
//   this->env->report_error(msg, sp);
// }

// void Typechecker::fatal_error(const std::string &msg, Span sp) {
//   this->env->report_error(msg, sp);
//   throw FatalTypeError(
//       "internal error: this exception should"
//       "be handled and errors reported with Environment::display_errors\n" +
//       msg);
// }

// Type Typechecker::unify(const Type &t1, const Type &t2, Span sp) {
//   try {
//     return this->unifier->unify(t1, t2);
//   } catch (const dmlc::Error &e) {
//     std::stringstream ss;
//     ss << "Error unifying `";
//     ss << PrintType(env, t1, WrapWidth(40));
//     ss << "` and `";
//     ss << PrintType(env, t2, WrapWidth(40));
//     ss << "`: " << e.what();
//     this->fatal_error(ss.str(), sp);
//   }
// }

// // template<typename T, typename U>

// // Add safe dynamic Array downcast.
// // Add static upcast?

// // Add to type utils.
// Array<Type> type_parameters(const Type &t) {
//   Array<Type> params;
//   auto type = t;
//   const TypeQuantifierNode *ty_quant;
//   while ((ty_quant = type.as<TypeQuantifierNode>())) {
//     params.push_back(ty_quant->id);
//     type = ty_quant->boundType;
//   }

//   return params;
// }

// template <typename I, typename F>
// Array<Param> ArrayMap(const Array<I> &data, F f) {
//   // probably a way to use std::transform.
//   Array<Param> output;
//   for (const I &el : data) {
//     output.push_back(f(el));
//   }
//   return output;
// }

// // There are some important questions around generalization
// // that we need to answer.
// Expr generalize(const Environment &env, const Expr &e) {
//   if (auto fn_node = e.as<FunctionNode>()) {
//     Typechecker tc(env);
//     auto ty = tc.VisitFunction(GetRef<Function>(fn_node), true);
//     auto ty_params = type_parameters(ty);
//     auto params = ArrayMap(fn_node->params, [&](const Param &p) {
//       return ParamNode::make(p->id, tc.resolve(p->type));
//     });
//     auto body = tc.resolve(fn_node->body);
//     auto ret_type = tc.resolve(fn_node->ret_type);
//     auto fn = FunctionNode::make(ty_params, params, ret_type, body);
//     // we should check in empty context to ensure typing is preserved.
//     // check(env, fn);
//     return fn;
//   } else {
//     throw dmlc::Error("can only apply generalize to a function.");
//   }
// }

TVM_REGISTER_API("relay._type_infer.check_expr")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Environment env = args[0];
      Expr e = args[1];
      *ret = check(env, e);
    });

// TVM_REGISTER_API("relay._tyck.check_item")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Environment env = args[0];
//       Item i = args[1];
//       *ret = check(env, i);
//     });

TVM_REGISTER_API("relay._type_infer._get_checked_type")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Expr e = args[0];
      *ret = e->checked_type();
    });

// TVM_REGISTER_API("relay._tyck.generalize")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       *ret = generalize(args[0], args[1]);
//     });

IncompleteType IncompleteTypeNode::make(TypeParamNode::Kind kind) {
  std::shared_ptr<IncompleteTypeNode> n = std::make_shared<IncompleteTypeNode>();
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
