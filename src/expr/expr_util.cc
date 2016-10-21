/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr_util.cc
 */
#include <tvm/expr_util.h>
#include <tvm/op.h>

namespace tvm {

inline bool is_ingeter(DataType t) {
  return t == kInt32;
}

/*! \brief Canonical form of expression */
struct CanonicalExpr {
  /*! \brief the e->value */
  std::unordered_map<Expr, int64_t> dict;
  /*! \brief constant value in the expresssion */
  int64_t constant{0};
  // change CanonicalExpr as expr
  inline Expr AsExpr() const {
    Expr e;
    using KV = std::pair<Expr, int64_t>;
    std::vector<KV> tlist(dict.begin(), dict.end());
    std::sort(tlist.begin(), tlist.end(), [](const KV& lhs, const KV& rhs) {
        return lhs.first.hash() < rhs.first.hash();
      });
    for (auto &kv : tlist) {
      if (kv.second == 0) continue;
      Expr tmp;
      if (kv.second == 1) {
        tmp = kv.first;
      } else {
        tmp = kv.first * kv.second;
      }
      if (e.is_null()) {
        e = tmp;
      } else {
        e = e + tmp;
      }
    }
    if (e.is_null()) {
      return IntConstant(constant);
    } else {
      if (constant != 0) e = e + constant;
      return e;
    }
  }

  inline void Add(const Expr& e, int beta) {
    auto it = dict.find(e);
    if (it != dict.end()) {
      it->second += beta;
      if (it->second == 0) dict.erase(it);
    } else {
      dict[e] = beta;
    }
  }
};

//  out += beta * Canonicalize(e)
void AddCanonical(const Expr& e,
                  CanonicalExpr* out,
                  int beta) {
  static const BinaryOp* add_op = BinaryOp::Get("+");
  static const BinaryOp* sub_op = BinaryOp::Get("-");
  static const BinaryOp* mul_op = BinaryOp::Get("*");
  static const BinaryOp* max_op = BinaryOp::Get("max");
  static const BinaryOp* min_op = BinaryOp::Get("min");


  CHECK(!e.is_null()) << "cannot simplify null";
  switch (e.node_type()) {
    case kIntNode: {
      out->constant += (e.Get<IntNode>()->value) * beta; return;
    }
    case kBinaryOpNode: {
      const auto* n = e.Get<BinaryOpNode>();
      if (n->op == add_op) {
        AddCanonical(n->lhs, out, beta);
        AddCanonical(n->rhs, out, beta);
        return;
      }
      if (n->op == sub_op) {
        AddCanonical(n->lhs, out, beta);
        AddCanonical(n->rhs, out, -beta);
        return;
      }
      if (n->op == mul_op) {
        if (n->lhs.node_type() == kIntNode) {
          AddCanonical(n->rhs, out, beta * (n->lhs.Get<IntNode>()->value)); return;
        } else if (n->rhs.node_type() == kIntNode) {
          AddCanonical(n->lhs, out, beta * (n->rhs.Get<IntNode>()->value)); return;
        }
        CanonicalExpr clhs, crhs;
        AddCanonical(n->lhs, &clhs, 1);
        if (clhs.dict.size() == 0) {
          AddCanonical(n->rhs, out, beta * clhs.constant); return;
        }
        AddCanonical(n->rhs, &crhs, 1);
        if (crhs.dict.size() == 0) {
          AddCanonical(n->lhs, out, beta * crhs.constant); return;
        }
        out->Add(e, beta); return;
      }
      if (n->op == max_op) {
        CanonicalExpr res;
        AddCanonical(n->lhs, &res, 1);
        AddCanonical(n->rhs, &res, -1);
        if (res.dict.size() == 0) {
          if (res.constant > 0) {
            AddCanonical(n->lhs, out, beta); return;
          } else {
            AddCanonical(n->rhs, out, beta); return;
          }
        } else {
          out->Add(e, beta); return;
        }
      }
      if (n->op == min_op) {
        CanonicalExpr res;
        AddCanonical(n->lhs, &res, 1);
        AddCanonical(n->rhs, &res, -1);
        if (res.dict.size() == 0) {
          if (res.constant <= 0) {
            AddCanonical(n->lhs, out, beta); return;
          } else {
            AddCanonical(n->rhs, out, beta); return;
          }
        } else {
          out->Add(e, beta); return;
        }
      }
      out->Add(e, beta);
      return;
    }
    default: {
      out->Add(e, beta); return;
    }
  }
}

Expr Simplify(Expr src) {
  CanonicalExpr cexpr;
  AddCanonical(src, &cexpr, 1);
  return cexpr.AsExpr();
}

void Expr::Print(std::ostream& os) const {
  if (is_null()) {
    os << "null"; return;
  }
  switch (this->node_type()) {
    case kVarNode: {
      os << Get<VarNode>()->name; return;
    }
    case kIntNode: {
      os << Get<IntNode>()->value; return;
    }
    case kFloatNode: {
      os << Get<FloatNode>()->value; return;
    }
    case kBinaryOpNode: {
      const auto* n = Get<BinaryOpNode>();
      const char* fname = n->op->FunctionName();
      if (fname[1] == '\0' && !isalpha(fname[0])) {
        os << '(';
        n->lhs.Print(os);
        os << ' ' << fname[0] << ' ';
        n->rhs.Print(os);
        os << ')';
      } else {
        os << fname << '(';
        n->lhs.Print(os);
        os << ", ";
        n->rhs.Print(os);
        os << ')';
      }
      return;
    }
    case kUnaryOpNode: {
      const auto* n = Get<UnaryOpNode>();
      os << n->op->FunctionName() << '(';
      n->src.Print(os);
      os << ')';
      return;
    }
    case kReduceNode: {
      const auto* n = Get<ReduceNode>();
      os << "reduce("<< n->op->FunctionName() << ", ";
      n->src.Print(os);
      os << ", " << n->rdom << ')';
      return;
    }
    case kTensorReadNode: {
      const auto* n = Get<TensorReadNode>();
      os << n->tensor.name() << n->indices;
      return;
    }
    default: {
      LOG(FATAL) << "not able to handle type " << typeid(node_.get()).name();
    }
  }
}


}  // namespace tvm
