#ifndef TOPI_DETAIL_BROADCAST_H
#define TOPI_DETAIL_BROADCAST_H

#include <tvm/tvm.h>
#include <tvm/ir_pass.h>

namespace topi { namespace detail {

struct BroadcastHelper {
  std::deque<tvm::Expr> commonShape;
  std::deque<tvm::Var> allVars;
  std::deque<tvm::Var> vars1;
  std::deque<tvm::Var> vars2;
};

inline BroadcastHelper broadcastShape(const tvm::Array<tvm::Expr>& shape1,
                                      const tvm::Array<tvm::Expr>& shape2) {
  BroadcastHelper bh;
  int s1Size = shape1.size();
  int s2Size = shape2.size();
  tvm::Expr one(1);
  int i;
  for (i = 1; i <= std::min(s1Size, s2Size); ++i) {
    bh.allVars.push_front(tvm::Var());
    if (tvm::ir::Equal(shape1[s1Size - i], shape2[s2Size - i])) {
      bh.commonShape.push_front(shape1[s1Size - i]);
      bh.vars1.push_front(bh.allVars[0]);
      bh.vars2.push_front(bh.allVars[0]);
    } else if (tvm::ir::Equal(one, shape1[s1Size - i])) {
      CHECK(! tvm::ir::Equal(one, shape2[s2Size - i]));
      bh.commonShape.push_front(shape2[s2Size - i]);
      bh.vars2.push_front(bh.allVars[0]);
    } else if (tvm::ir::Equal(one, shape2[s2Size - i])) {
      bh.commonShape.push_front(shape1[s1Size - i]);
      bh.vars1.push_front(bh.allVars[0]);
    } else {
      CHECK(false) <<
        "Incompatible broadcast dims: " <<
        shape1[s1Size - i] << " and " << shape2[s2Size - i] << " in: " <<
        tvm::Array<tvm::Expr>(shape1.begin(), shape1.end()) << " and " <<
        tvm::Array<tvm::Expr>(shape2.begin(), shape2.end());
    }
  }
  // Remaining dimensions whether on shape1 or shape2 can always be completed
  auto maxSize = std::max(s1Size, s2Size);
  auto& shape = (s1Size > s2Size) ? shape1 : shape2;
  auto& vars = (s1Size > s2Size) ? bh.vars1 : bh.vars2;
  for (i = i; i <= maxSize; ++i) {
    bh.allVars.push_front(tvm::Var());
    bh.commonShape.push_front(shape[maxSize - i]);
    vars.push_front(bh.allVars[0]);
  }
  return bh;
}

inline tvm::Array<tvm::Expr> inputShapeFromBroadcast(
    const tvm::Array<tvm::Var>& ovars,
    const tvm::Tensor& T,
    const std::deque<tvm::Var>& myVars,
    const std::deque<tvm::Var>& allVars) {
  tvm::Array<tvm::Expr> ivars;
  CHECK_EQ(ovars.size(), allVars.size());
  // N^2, could use a map but NBD..
  int expectedDims = T->shape.size();
  for (int i = 0; i < ovars.size(); ++i) {
    bool found = false;
    for (int j = 0; j < myVars.size(); ++j) {
      if (tvm::ir::Equal(allVars[i], myVars[j])) {
        ivars.push_back(ovars[i]);
        found = true;
        break;
      }
    }
    // Only inject 0 here if we have not yet reached the dimension of I
    // (i.e. this must be a 1)
    if (!found && (ovars.size() - i) <= expectedDims) {
      ivars.push_back(tvm::Expr(0));
    }
  }
  CHECK(expectedDims == ivars.size());
  return ivars;
}

typedef std::function<tvm::Expr(tvm::Expr, tvm::Expr)> BinaryExpr;

inline tvm::Tensor withBroadcast(BinaryExpr op,
                                 const tvm::Tensor& A,
                                 const tvm::Tensor& B) {
  auto bh = broadcastShape(A->shape, B->shape);
  auto l = [&](tvm::Array<tvm::Var> ovars) {
    return op(
      A(inputShapeFromBroadcast(ovars, A, bh.vars1, bh.allVars)),
      B(inputShapeFromBroadcast(ovars, B, bh.vars2, bh.allVars))
    );
  };
  return tvm::compute(
    tvm::Array<tvm::Expr>(bh.commonShape.begin(), bh.commonShape.end()), l);
}

}} // ns topi::detail


#endif // TOPI_DETAIL_BROADCAST_H
