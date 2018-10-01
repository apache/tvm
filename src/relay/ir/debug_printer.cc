/*!
 *  Copyright (c) 2018 by Contributors
 * \file debug_printer.cc
 * \brief A pretty printer for the Relay IR.
 * As we had not determined a formal syntax yet, right now it is only for debug purpose.
 */

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/environment.h>
#include <tvm/relay/error.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include "../pass/type_functor.h"
#include "doc.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

Doc KindDocify(TypeParamNode::Kind k) {
  switch (k) {
    case TypeParamNode::kShapeVar:
      return DocOfStr("ShapeVar");
    case TypeParamNode::kShape:
      return DocOfStr("Shape");
    case TypeParamNode::kBaseType:
      return DocOfStr("BaseType");
    case TypeParamNode::kType:
      return DocOfStr("Type");
    default:
      LOG(FATAL) << "unreachable code: case not handle in kind";
      throw;  // log fatal throw but compiler doesnt know
  }
}

template<typename T>
std::vector<Doc> MapDocify(const tvm::Array<T>& arr, const std::function<Doc(const T&)>& f) {
  std::vector<Doc> vec;
  for (size_t i = 0; i < arr.size(); ++i) {
    vec.push_back(f(arr[i]));
  }
  return vec;
}

template<typename T, typename Hash = std::hash<T>, typename Eq = std::equal_to<T>>
class Counter {
  std::unordered_map<T, size_t, Hash, Eq> cnt_;

 public:
  Counter() = default;
  Counter(const Counter&) = delete;
  size_t operator()(const T& t) {
    auto v = cnt_.count(t) == 0 ? 0 : cnt_.at(t) + 1;
    cnt_[t] = v;
    return v;
  }
};

std::string Mangle(const std::string& str, size_t s) {
  return str + "_" + std::to_string(s);
  // return s == 0 ? str : str + "_" + std::to_string(s - 1);
  // the above line look prettier but is dangerous:
  // suppose we have x, x, x_0. mangling will give x, x_0, x_0!
  // the save approach give x_0, x_1, x_0_1, and in fact never clash:
  // stripping _([0-9]*) is invert of mangle under all circumstances.
  // another problem is we need to prevent Var/TypeParam/GlobalVar clashing each other.
}

constexpr size_t indent = 2;

struct TypeParamName {
  bool operator==(const TypeParamName&) const {
    return true;
  }
};

struct mhash {
  size_t operator()(const ::tvm::relay::TypeParamName&) const noexcept {
    return 0;
  }
};

class TypeDocifier : private TypeFunctor<Doc(const Type& n)> {
  Environment env;
  Counter<TypeParamName, mhash> cnt;
  std::unordered_map<TypeParam, Doc, NodeHash, NodeEqual> map;

  std::vector<Doc> DocifyTypeArray(const tvm::Array<Type>& arr) {
    return MapDocify<Type>(arr, [=](const Type& t) { return Docify(t); });
  }

  std::vector<Doc> DocifyTypeParam(const tvm::Array<TypeParam>& arr) {
    return MapDocify<TypeParam>(arr, [=](const TypeParam& tp) { return Docify(tp); });
  }

  std::vector<Doc> DocifyTypeConstraint(const tvm::Array<TypeConstraint>& arr) {
    return MapDocify<TypeConstraint>(arr, [=](const TypeConstraint& tc) { return Docify(tc); });
  }

  Doc VisitType_(const TensorTypeNode* t) final {
    return DocOfStr("tensor");
  }

  Doc VisitType_(const TypeParamNode* p) final {
    auto tp = GetRef<TypeParam>(p);
    if (map.count(tp) == 0) {
      auto name =
        DocOfStr(Mangle("tp", cnt(TypeParamName())) +
                 std::string(":")) +
        KindDocify(p->kind);
      map.insert(std::pair<TypeParam, Doc>(tp, name));
    }
    return map.at(tp);
  }

  Doc Quantify(const tvm::Array<TypeParam>& tp, const Doc& d) {
    if (tp.size() == 0) {
      return d;
    }
    return Seq("forall", DocifyTypeParam(tp), ",") + Sep() + d;
  }

  Doc Constraint(const tvm::Array<TypeConstraint>& tc, const Doc& d) {
    if (tc.size() == 0) {
      return d;
    }
    return Seq("(", DocifyTypeConstraint(tc), ") =>") + Sep() + d;
  }

  Doc VisitType_(const FuncTypeNode* f) final {
    auto inner = Seq("<", DocifyTypeArray(f->arg_types), ">") + Sep() +
                 DocOfStr("->") + Sep() + Docify(f->ret_type);
    return Group(Quantify(f->type_params,
                          Constraint(f->type_constraints, inner)));
  }

  Doc VisitType_(const TypeRelationNode* r) final {
    return DocOfStr("Relation") + Seq("(", DocifyTypeArray(r->args), ")");
  }

  Doc VisitType_(const TupleTypeNode* t) final {
    return Seq("<", DocifyTypeArray(t->fields), ">");
  }

  Doc VisitType_(const IncompleteTypeNode* i) final {
    return DocOfStr("_");
  }

 public:
  TypeDocifier(const Environment& env) : env(env) { }

  Doc Docify(const Type& t) { return t.get() ? (*this)(t) : DocOfStr("_"); }
};

class ExprDocifier : private ExprFunctor<Doc(const Expr& n)> {
  Environment env;
  Counter<std::string> cnt;
  std::unordered_map<Var, std::string, NodeHash, NodeEqual> map;
  TypeDocifier td;

  std::string VarName(const Var& v) {
    if (map.count(v) == 0) {
      map.insert(std::pair<Var, std::string>(v, Mangle(v->name_hint, cnt(v->name_hint))));
    }
    return map.at(v);
  }

  Doc TypeAnnotation(const Doc& d, const Type& t) {
    // test for t being null. probably shouldnt has null. should talk to jared.
    if (!t.get() || t.as<IncompleteTypeNode>()) {
      return d;
    } else {
      return d + DocOfStr(":") + td.Docify(t);
    }
  }

  std::vector<Doc> DocifyExprArray(const tvm::Array<Expr>& arr) {
    std::vector<Doc> vec;
    for (size_t i = 0; i < arr.size(); ++i) {
      vec.push_back(Docify(arr[i]));
    }
    return vec;
  }

  std::vector<Doc> DocifyParamArray(const tvm::Array<Param>& arr) {
    std::vector<Doc> vec;
    for (size_t i = 0; i < arr.size(); ++i) {
      vec.push_back(Docify(arr[i]));
    }
    return vec;
  }

  Doc VisitExpr_(const ConstantNode* c) final {
    return DocOfStr("some_constant");
  }

  Doc VisitExpr_(const TupleNode* t) final {
    return Seq("<", DocifyExprArray(t->fields), ">");
  }

  Doc VisitExpr_(const VarNode* v) final {
    return DocOfStr(VarName(GetRef<Var>(v)));
  }

  Doc VisitExpr_(const GlobalVarNode* g) final {
    return DocOfStr(g->name_hint);
  }

  Doc VisitExpr_(const ParamNode* p) final {
    return TypeAnnotation(Docify(p->var), p->type);
  }

  Doc VisitExpr_(const FunctionNode* f) final {
    return Group(TypeAnnotation(Seq("(", DocifyParamArray(f->params), ")"), f->ret_type) + Sep() +
                 DocOfStr("=>") + Sep() +
                 Block(indent, "{", Docify(f->body), "}"));
  }

  Doc VisitExpr_(const CallNode* c) final {
    auto args = DocifyExprArray(c->args);
    return Docify(c->op) + Seq("<", DocifyExprArray(c->args), ">");
  }

  Doc VisitExpr_(const LetNode* l) final {
    return Group(DocOfStr("let") + Sep() + TypeAnnotation(Docify(l->var), l->value_type) + Sep() +
                 DocOfStr("=") + Sep() + Docify(l->value) + DocOfStr(";") + Endl() +
                 Docify(l->body));
  }

  Doc VisitExpr_(const IfNode* i) final {
    return Group(DocOfStr("if") + Sep() + Docify(i->cond) + Sep() +
                 Block(indent, "{", Docify(i->true_branch), "}") + Sep() +
                 DocOfStr("else") + Sep() +
                 Block(indent, "{", Docify(i->false_branch), "}"));
  }

  Doc VisitExpr_(const OpNode* o) final {
    return DocOfStr(o->name);
  }

 public:
  ExprDocifier(const Environment& env) : env(env), td(env) { }

  Doc Docify(const Expr& e) { return (*this)(e); }
};

Doc DocOfExpr(const Environment& env, const Expr& expr) {
  ExprDocifier d(env);
  return d.Docify(expr);
}

Doc DocOfType(const Environment& env, const Type& expr) {
  TypeDocifier d(env);
  return d.Docify(expr);
}

RDoc ExprRDoc(const Environment& env, const Expr& expr) {
  return Layout(DocOfExpr(env, expr));
}

RDoc TypeRDoc(const Environment& env, const Type& expr) {
  return Layout(DocOfType(env, expr));
}

std::ostream & DebugPrint(const Environment& env, const Expr& e, std::ostream& os) {
  return os << ExprRDoc(env, e);
}

std::ostream & DebugPrint(const Environment& env, const Type& t, std::ostream& os) {
  return os << TypeRDoc(env, t);
}

std::string PrintExpr(const Environment& env, const Expr& e) {
  std::stringstream ss;
  ss << ExprRDoc(env, e);
  return ss.str();
}

std::string PrintType(const Environment& env, const Type& t) {
  std::stringstream ss;
  ss << TypeRDoc(env, t);
  return ss.str();
}

TVM_REGISTER_API("relay._expr._debug_print")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    NodeRef x = args[1];
    std::cout << x << std::endl;
    if (x.as<TypeNode>()) {
      *ret = PrintType(args[0], Downcast<Type>(x));
    } else {
      *ret = PrintExpr(args[0], Downcast<Expr>(x));
    }
  });

}  // namespace relay
}  // namespace tvm
