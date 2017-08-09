/*!
 *  Copyright (c) 2016 by Contributors
 * \file buffer.cc
 */
#include <tvm/buffer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <iterator>
#include "../arithmetic/compute_expr.h"

namespace tvm {

Array<Expr> GetStrides(Array<Expr> shape) {
  CHECK_NE(shape.size(), 0U);
  std::vector<Expr> vec{make_const(shape[0].type(), 1)};
  for (size_t i = shape.size() - 1; i != 0; --i) {
    vec.push_back(shape[i - 1] * vec.back());
  }
  return Array<Expr>(vec.rbegin(), vec.rend());
}

Buffer decl_buffer(Array<Expr> shape,
                   Type dtype,
                   std::string name) {
  return BufferNode::make(
      Var(name, Handle()),
      dtype,
      shape,
      Array<Expr>(),
      Expr(),
      name,
      "",
      0, 0);
}

// Split the given expression by the add operator
inline std::vector<Expr> _expr_add_split(const Expr &expr) {
  using namespace Halide::Internal;
  std::vector<Expr> ret;
  auto expr_add_match = expr.as<Add>();
  if (expr_add_match) {
    auto l_ret = _expr_add_split(expr_add_match->a);
    auto r_ret = _expr_add_split(expr_add_match->b);
    ret.insert(ret.end(),
               std::make_move_iterator(l_ret.begin()),
               std::make_move_iterator(l_ret.end()));
    ret.insert(ret.end(),
               std::make_move_iterator(r_ret.begin()),
               std::make_move_iterator(r_ret.end()));
  } else {
    ret.emplace_back(expr);
  }
  return ret;
}


// Searches for the following types of expr:
//   mult_expr = (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   mod_l_expr = c
//   mod_r_expr = k1 * k2 * ... * ki
// If it can be optimized, returns (true, (a1 + a2 + ... + aj) * kt * ... * ki + c)
// Currently the we will not search the add/mult combinations exhaustively
//   as it will take too much computation.
inline std::pair<bool, Expr> _merge_mul_mod(const Expr &mult_expr,
                                            const Expr &mod_l_expr,
                                            const Expr &mod_r_expr) {
  using namespace Halide::Internal;
  using namespace ir;
  auto mult_ptr = mult_expr.as<Mul>();
  if (!mult_ptr) return std::make_pair(false, Expr());
  Expr mult_outer = mult_ptr->b;
  const Expr* inner = &(mult_ptr->a);
  // 1. Calculate the outer multiplier
  while (1) {
    mult_ptr = inner->as<Mul>();
    if (mult_ptr) {
      inner = &(mult_ptr->a);
      mult_outer = mult_ptr->b * mult_outer;
    } else {
      break;
    }
  }
  // 2. Search for the pattern c / (...) * (...) + c % (...)
  // We match the search element with Add, Mul and Div.
  //   If Add is found, we need to continue our search for the rhs
  //   If Mult is found, we will expand the inner multiplication factor
  //   If Div is found, we will go on testing whether lhs matches the lhs of mod expr
  //      and returns the optimization result.
  const Expr* search_ptr = inner;
  Expr mult_inner;  // The inner multiplication factor
  bool has_mult_inner = false;  // whether mult_inner is empty
  Expr no_opt_sum;  // Sum of the exprs that cannot be optimized
  bool has_no_opt_sum = false;  // Whether no_opt_sum is empty
  while (1) {
    auto inner_div_ptr = search_ptr->as<Div>();
    auto inner_mult_ptr = search_ptr->as<Mul>();
    auto inner_add_ptr = search_ptr->as<Add>();
    if (!inner_div_ptr && !inner_mult_ptr && !inner_add_ptr) {
      return std::make_pair(false, Expr());
    } else if (inner_div_ptr) {
      Expr overall_mult = has_mult_inner ? mult_inner * mult_outer : mult_outer;
      if (Equal(overall_mult, inner_div_ptr->b)
          && Equal(overall_mult, mod_r_expr)
          && Equal(inner_div_ptr->a, mod_l_expr)) {
        // Found!
        Expr ret = has_no_opt_sum ? no_opt_sum * mult_outer + mod_l_expr : mod_l_expr;
        return std::make_pair(true, ret);
      } else {
        return std::make_pair(false, Expr());
      }
    } else if (inner_mult_ptr) {
      mult_inner = has_mult_inner ? inner_mult_ptr->b * mult_inner : inner_mult_ptr->b;
      search_ptr = &(inner_mult_ptr->a);
      has_mult_inner = true;
    } else if (inner_add_ptr) {
      if (has_mult_inner) {
        return std::make_pair(false, Expr());
      }
      no_opt_sum = has_no_opt_sum ? no_opt_sum + inner_add_ptr->a : inner_add_ptr->a;
      has_no_opt_sum = true;
      search_ptr = &(inner_add_ptr->b);
    } else {
      LOG(FATAL) << "Unexpected search result!";
      break;
    }
  }
  return std::make_pair(false, Expr());
}

// Insert the elements into the corresponding mult_exprs and mod_exprs.
// If the element is found to match Mul, it will be pushed to the mult_exprs.
// If the element it found to match Mod, it will be pused to the mod_exprs.
// Otherwise, the elements will be added to the no_opt_sum variable
inline void _merge_mul_mod_insert_eles(const std::vector<Expr>& eles,
                                       std::list<Expr>* mult_exprs,
                                       std::list<std::pair<Expr, Expr> >* mod_exprs,
                                       Expr* no_opt_sum,
                                       bool* has_no_opt_sum,
                                       bool* has_mult,
                                       bool* has_mod) {
  using namespace Halide::Internal;
  using namespace ir;
  *has_mult = false;
  *has_mod = false;
  for (const Expr& ele : eles) {
    auto mod_ptr = ele.as<Mod>();
    auto mult_ptr = ele.as<Mul>();
    if (mod_ptr) {
      *has_mod = true;
      mod_exprs->emplace_back(std::make_pair(std::move(mod_ptr->a), std::move(mod_ptr->b)));
    } else if (mult_ptr) {
      *has_mult = true;
      mult_exprs->emplace_back(ele);
    } else {
      *no_opt_sum = *has_no_opt_sum ? *no_opt_sum + ele : ele;
      *has_no_opt_sum = true;
    }
  }
}

// Searches for this types of expr:
//   (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   + c % (k1 * k2 * ... * ki)
// and simplifies to (a1 + a2 + ... + aj) * kt * ... * ki + c
// The search will be performed repeatively until no pattern is found.
// Return: a pair with (false, Expr()) if cannot be optimized.
//         a pair with (true, optimized_expr) if can be optimized
inline Expr opt_merge_mul_mod(const Expr &base) {
  using namespace Halide::Internal;
  using namespace ir;
  // 1. Prepare the lists.
  // We store two lists, a list that contain all the elements that match Mul and
  //                     a list that contain all the elements that match Mod.
  // The elements in the Mod will be used to match against the elements in Mul.
  // The result will then be split and pushed back to these two lists.
  std::vector<Expr> eles = _expr_add_split(base);
  std::list<Expr> mult_exprs;
  std::list<std::pair<Expr, Expr> > mod_exprs;
  Expr no_opt_sum;
  bool has_no_opt_sum = false;
  bool has_mult;
  bool has_mod;
  _merge_mul_mod_insert_eles(eles, &mult_exprs, &mod_exprs,
                             &no_opt_sum, &has_no_opt_sum, &has_mult, &has_mod);
  bool find_opt = false;
  std::list<std::pair<Expr, Expr> >::iterator search_mod_it = mod_exprs.begin();
  // 2. Exhaustive Search
  while (search_mod_it != mod_exprs.end()) {
    std::list<Expr>::iterator mult_it = mult_exprs.begin();
    bool inner_find_opt = false;
    while (mult_it != mult_exprs.end()) {
      std::pair<bool, Expr> ret = _merge_mul_mod(*mult_it,
                                                 search_mod_it->first,
                                                 search_mod_it->second);
      if (ret.first) {
        inner_find_opt = true;
        auto temp_mod_it = search_mod_it;
        ++search_mod_it;
        mod_exprs.erase(temp_mod_it);
        mult_exprs.erase(mult_it);
        std::vector<Expr> ret_eles = _expr_add_split(ret.second);
        _merge_mul_mod_insert_eles(ret_eles, &mult_exprs, &mod_exprs,
                                   &no_opt_sum, &has_no_opt_sum, &has_mult, &has_mod);
        if (has_mult) {
          search_mod_it = mod_exprs.begin();
        }
        break;
      } else {
        ++mult_it;
      }
    }
    find_opt = find_opt || inner_find_opt;
    if (!inner_find_opt) {
      ++search_mod_it;
    }
  }
  if (!find_opt) {
    return base;
  }
  for (const Expr& ele : mult_exprs) {
    no_opt_sum = has_no_opt_sum ? no_opt_sum + ele : ele;
  }
  for (const std::pair<Expr, Expr>& ele : mod_exprs) {
    no_opt_sum = has_no_opt_sum ? no_opt_sum + ele.first % ele.second : ele.first % ele.second;
  }
  return no_opt_sum;
}

// The buffer offset in convention of number of elements of
// original data ignoring number of lanes.
// We also perform optimization to simplify the indexing expression.
inline Expr ElemOffset(const BufferNode* n, Array<Expr> index) {
  using namespace Halide::Internal;
  Expr base = n->elem_offset;
  if (n->strides.size() == 0) {
    CHECK_EQ(n->shape.size(), index.size());
    if (is_zero(base)) {
      base = index[0];
    } else {
      base = base + index[0];
    }
    base = opt_merge_mul_mod(base);
    for (size_t i = 1; i < index.size(); ++i) {
      base = opt_merge_mul_mod(base * n->shape[i] + index[i]);
    }
  } else {
    CHECK_EQ(n->strides.size(), index.size());
    if (is_zero(base)) {
      base = index[0] * n->strides[0];
    } else {
      base = base + index[0] * n->strides[0];
    }
    for (size_t i = 1; i < index.size(); ++i) {
      base = base + index[i] * n->strides[i];
    }
  }
  return base;
}

inline Expr BufferOffset(const BufferNode* n, Array<Expr> index, Type dtype) {
  Expr offset = ElemOffset(n, index);
  if (n->dtype.lanes() != 1) {
    offset = offset * make_const(offset.type(), dtype.lanes());
  }
  if (dtype.lanes() != 1) {
    return ir::Ramp::make(offset, make_const(offset.type(), 1), dtype.lanes());
  } else {
    return offset;
  }
}

Expr Buffer::vload(Array<Expr> begin, Type dtype) const {
  const BufferNode* n = operator->();
  CHECK(dtype.element_of() == n->dtype.element_of() &&
        dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot load " << dtype
      << " from buffer of " << n->dtype;
  return ir::Load::make(
      dtype, n->data, BufferOffset(n, begin, dtype),
      const_true(dtype.lanes()));
}

Stmt Buffer::vstore(Array<Expr> begin, Expr value) const {
  const BufferNode* n = operator->();
  Type dtype = value.type();
  CHECK(dtype.element_of() == n->dtype.element_of() &&
        dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot load " << dtype
      << " from buffer of " << n->dtype;
  return ir::Store::make(n->data, value, BufferOffset(n, begin, dtype),
                         const_true(dtype.lanes()));
}

Buffer Buffer::MakeStrideView() const {
  if ((*this)->strides.size() != 0) return *this;
  std::vector<Expr> temp;
  auto n = std::make_shared<BufferNode>(*operator->());
  Expr acc = make_const(n->shape[0].type(), 1);
  for (size_t i = n->shape.size(); i != 0 ; --i) {
    temp.push_back(acc);
    acc = acc * n->shape[i - 1];
  }
  for (size_t i = temp.size(); i != 0; --i) {
    n->strides.push_back(temp[i - 1]);
  }
  return Buffer(n);
}

Buffer Buffer::MakeSlice(Array<Expr> begins, Array<Expr> extents) const {
  const BufferNode* n = operator->();
  Expr elem_offset = ir::Simplify(ElemOffset(n, begins));
  Array<Expr> strides = n->strides;
  if (strides.size() == 0) {
    bool can_relax = true;
    bool need_stride = false;
    // check if stride is needed.
    for (size_t i = 0; i < extents.size(); ++i) {
      if (!can_relax) {
        if (!is_zero(begins[i]) ||
            !is_zero(ir::Simplify(extents[i] - n->shape[i]))) {
          need_stride = true;
        }
      }
      if (!is_one(extents[i])) can_relax = false;
    }
    // make stride.
    if (need_stride) {
      return MakeStrideView().MakeSlice(begins, extents);
    }
  }
  return BufferNode::make(n->data,
                          n->dtype,
                          extents,
                          strides,
                          elem_offset,
                          n->name + "_slice",
                          n->scope,
                          n->data_alignment,
                          0);
}

Expr Buffer::access_ptr(int access_mask, Type ptr_type) const {
  const BufferNode* self = operator->();
  Expr e_dtype = make_zero(self->dtype);
  Expr extent = (self->strides.size() == self->shape.size() ?
                 arith::ComputeExpr<ir::Mul>(self->strides[0], self->shape[0]):
                 arith::ComputeReduce<ir::Mul>(self->shape));
  Array<Expr> acc_args{
    e_dtype, self->data, self->elem_offset,
        extent, make_const(Int(32), access_mask)};
  return ir::Call::make(
      ptr_type, ir::intrinsic::tvm_access_ptr, acc_args, ir::Call::Intrinsic);
}

Buffer BufferNode::make(Var data,
                        Type dtype,
                        Array<Expr> shape,
                        Array<Expr> strides,
                        Expr elem_offset,
                        std::string name,
                        std::string scope,
                        int data_alignment,
                        int offset_factor) {
  auto n = std::make_shared<BufferNode>();
  n->data = std::move(data);
  n->dtype = dtype;
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->name = std::move(name);
  if (scope.length() == 0) {
    scope = "global";
  }
  n->scope = std::move(scope);
  if (!elem_offset.defined()) {
    elem_offset = make_const(n->shape[0].type(), 0);
  }
  if (data_alignment <= 0) {
    data_alignment = runtime::kAllocAlignment;
  }
  if (offset_factor == 0) {
    offset_factor = 1;
  }
  n->elem_offset = std::move(elem_offset);
  n->data_alignment = data_alignment;
  n->offset_factor = offset_factor;
  return Buffer(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferNode>([](const BufferNode *op, IRPrinter *p) {
    p->stream << "buffer(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(BufferNode);

}  // namespace tvm
