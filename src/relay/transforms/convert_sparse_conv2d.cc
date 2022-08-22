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
 * \file convert_sparse_conv2d.cc
 *
 * \brief Mutate conv2d operator to sparse conv2d operator
 */
#include <tvm/ir/expr.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relay {

// Search conv2d op weight name from Expr
class Conv2dOpWeightVisitor : private ExprVisitor {
 public:
  Conv2dOpWeightVisitor() : conv2d_op_(Op::Get("nn.conv2d")) {}

  Array<String> Search(const Expr& expr) {
    VisitExpr(expr);
    return memo_;
  }

 private:
  void VisitExpr_(const CallNode* n) final {
    if (n->op == conv2d_op_) {
      const auto weight = n->args[1].as<VarNode>();
      if (weight) {
        memo_.push_back(weight->name_hint());
      }
    }
    for (const auto& arg : n->args) {
      VisitExpr(arg);
    }
  }
  // Cache op
  const Op& conv2d_op_;

  Array<String> memo_;
};  // SearchConv2dOpWeight

Array<String> SearchConv2dOpWeight(const Expr& e) { return Conv2dOpWeightVisitor().Search(e); }

TVM_REGISTER_GLOBAL("relay.analysis.search_conv2d_op_weight").set_body_typed(SearchConv2dOpWeight);

// Mutate ```nn.conv2d``` to ```nn.sparse_conv2d```
class Conv2dToSparseConv2dMutator : public ExprRewriter {
 public:
  Conv2dToSparseConv2dMutator(const Array<ObjectRef>& weight_name,
                              const Array<Array<PrimExpr>>& weight_shape, const String& layout,
                              int kernel_size)
      : conv2d_op_(Op::Get("nn.conv2d")), sparse_conv2d_op_(Op::Get("nn.sparse_conv2d")) {
    ICHECK_EQ(weight_name.size(), weight_shape.size());
    layout_ = layout;
    kernel_size_ = kernel_size;
    for (size_t i = 0; i < weight_name.size(); ++i) {
      ICHECK(weight_name[i]->IsInstance<runtime::StringObj>());
      std::string k = weight_name[i].as<runtime::StringObj>()->data;
      const auto& ws = weight_shape[i];
      std::vector<int> v(ws.size());
      for (size_t j = 0; j < ws.size(); ++j) {
        v[j] = ws[j].as<IntImmNode>()->value;
      }
      target_weights_.emplace(k, v);
    }
  }

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (pre->op == conv2d_op_) {
      const auto weight = pre->args[1].as<VarNode>();
      if (weight) {
        if (target_weights_.count(weight->name_hint())) {
          const auto& prefix = weight->name_hint();
          const auto& ws = target_weights_.at(prefix);
          const auto data = post.as<CallNode>()->args[0];
          relay::TensorType ws_data_type, ws_indices_type, ws_indptr_type;
          if (ws.size() == 5) {
            ws_data_type = relay::TensorType({ws.at(0), ws.at(1), ws.at(2)}, DataType::Float(32));
            ws_indices_type = relay::TensorType({ws.at(3)}, DataType::Int(32));
            ws_indptr_type = relay::TensorType({ws.at(4)}, DataType::Int(32));
          } else if (ws.size() == 4) {
            ws_data_type = relay::TensorType({ws.at(0), ws.at(1)}, DataType::Float(32));
            ws_indices_type = relay::TensorType({ws.at(2)}, DataType::Int(32));
            ws_indptr_type = relay::TensorType({ws.at(3)}, DataType::Int(32));
          }
          Var weight_data(prefix + ".data", ws_data_type);
          Var weight_indices(prefix + ".indices", ws_indices_type);
          Var weight_indptr(prefix + ".indptr", ws_indptr_type);
          auto attrs = make_object<SparseConv2DAttrs>();
          attrs->layout = std::move(layout_);
          attrs->kernel_size = Array<IndexExpr>{kernel_size_, kernel_size_};
          return Call(sparse_conv2d_op_, {data, weight_data, weight_indices, weight_indptr},
                      Attrs(attrs));
        }
      }
    }
    return post;
  }

 private:
  // Cached op
  const Op& conv2d_op_;
  const Op& sparse_conv2d_op_;
  std::unordered_map<std::string, std::vector<int>> target_weights_;
  String layout_;
  int kernel_size_;
};  // class Conv2dToSparseConv2dAlter

Expr Conv2dToSparse(const Expr& e, const Array<ObjectRef>& weight_name,
                    const Array<Array<PrimExpr>>& weight_shape, const String& layout,
                    int kernel_size) {
  auto rewriter = Conv2dToSparseConv2dMutator(weight_name, weight_shape, layout, kernel_size);
  return PostOrderRewrite(e, &rewriter);
}

template <typename elemTy, size_t... Is>
auto unpack_to_tuple_internal(elemTy* arr, std::index_sequence<Is...>) {
  return std::make_tuple(arr[Is]...);
}

template <int N, typename elemTy>
auto unpack_to_tuple(elemTy* arr) {
  return unpack_to_tuple_internal(arr, std::make_index_sequence<N>{});
}

struct Range {
  size_t dim;
  explicit Range(size_t d) : dim(d) {}

  struct iterpoint {
    size_t val, lim;
    iterpoint(size_t v1, size_t v2) : val(v1), lim(v2) {}

    size_t operator*() const { return val; }

    iterpoint operator/(const iterpoint& rhs) const {
      return iterpoint(val * rhs.lim + rhs.val, lim * rhs.lim);
    }
  };

  struct iterator {
    size_t val, lim;
    iterator(size_t v1, size_t v2) : val(v1), lim(v2) {}

    bool operator!=(const iterator& rhs) const { return val != rhs.val; }

    void operator++() { ++val; }

    iterpoint operator*() const { return iterpoint(val, lim); }
  };

  iterator begin() { return iterator(0, dim); }

  iterator end() { return iterator(dim, dim); }
};

// Mutate ```nn.conv2d``` to ```nn.sparse_conv2d```
class Conv2dToSparseConv2dMutator2 : public ExprRewriter {
 public:
  Conv2dToSparseConv2dMutator2(const String& layout, int kernel_size, int blockH, int blockW,
                               double sparse_thresh)
      : sparse_conv2d_op_(Op::Get("nn.sparse_conv2d")),
        dev_cpu0_{DLDeviceType::kDLCPU, 0},
        layout_(layout),
        kernel_size_(kernel_size),
        blockH_(blockH),
        blockW_(blockW),
        sparse_thresh_(sparse_thresh) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    // check op type & attrs
    const auto pre_attrs = pre->attrs.as<Conv2DAttrs>();
    if (!pre_attrs || pre_attrs->data_layout != layout_ ||
        pre_attrs->strides[0].as<IntImmNode>()->value != 1 ||
        pre_attrs->kernel_size[0].as<IntImmNode>()->value != kernel_size_)
      return post;
    // check constant weight
    const auto pre_weight_node = pre->args[1].as<ConstantNode>();
    if (!pre_weight_node) return post;

    // check weight dtype & shape
    auto&& pre_weight = pre_weight_node->data;
    auto dtype = pre_weight.DataType(), itype = runtime::DataType::Int(32);
    ICHECK(dtype.code() == DataType::kFloat && dtype.bits() == 32);  // float32 only
    auto pre_weight_shape = unpack_to_tuple<4>(pre_weight.Shape().data());
    int O, I, H, W;
    if (layout_ == "NCHW") {
      std::tie(O, I, H, W) = pre_weight_shape;
    } else {  // NHWC
      std::tie(H, W, I, O) = pre_weight_shape;
    }
    int CO = O, CI = H * W * I;

    // copy to vector
    std::vector<float> pre_weight_data(CO * CI);
    pre_weight.CopyToBytes(pre_weight_data.data(), pre_weight_data.size() * sizeof(float));
    if (layout_ == "NHWC") {
      std::vector<float> tmp(pre_weight_data.size());
      for (auto i : Range(CO))
        for (auto j : Range(CI)) tmp[*(i / j)] = pre_weight_data[*(j / i)];
      std::swap(tmp, pre_weight_data);
    }
    // convert to BSR
    std::vector<float> wdata, block(blockH_ * blockW_);
    std::vector<int32_t> windices, windptr;
    for (auto bh : Range(CO / blockH_)) {
      windptr.push_back(windices.size());
      for (auto bw : Range(CI / blockW_)) {
        int cntnnz = 0;
        for (auto i : Range(blockH_))
          for (auto j : Range(blockW_)) {
            auto tmp = pre_weight_data[*(bh / i / bw / j)];
            if (tmp) cntnnz++;
            block[*(i / j)] = tmp;
          }
        if (cntnnz) {
          wdata.insert(wdata.end(), block.begin(), block.end());
          windices.push_back(*bw);
        }
      }
    }
    windptr.push_back(windices.size());
    double sprate = 1 - 1.0 * wdata.size() / pre_weight_data.size();
    if (sprate < sparse_thresh_) return post;

    // constrct return data
    int nnz = windices.size();
    auto weight_data = runtime::NDArray::Empty({nnz, blockH_, blockW_}, dtype, dev_cpu0_);
    auto weight_indices = runtime::NDArray::Empty({nnz}, itype, dev_cpu0_);
    auto weight_indptr = runtime::NDArray::Empty({CO / blockH_ + 1}, itype, dev_cpu0_);
    weight_data.CopyFromBytes(wdata.data(), wdata.size() * sizeof(float));
    weight_indices.CopyFromBytes(windices.data(), windices.size() * sizeof(int32_t));
    weight_indptr.CopyFromBytes(windptr.data(), windptr.size() * sizeof(int32_t));

    // construct return call
    auto args = runtime::Array<relay::Expr>{post.as<CallNode>()->args[0], Constant(weight_data),
                                            Constant(weight_indices), Constant(weight_indptr)};
    auto attrs = make_object<SparseConv2DAttrs>();
    attrs->layout = layout_;
    attrs->kernel_size = Array<IndexExpr>{kernel_size_, kernel_size_};
    return Call(sparse_conv2d_op_, args, Attrs(attrs));
  }

 private:
  const Op& sparse_conv2d_op_;
  DLDevice dev_cpu0_;
  String layout_;
  int kernel_size_, blockH_, blockW_;
  double sparse_thresh_;
};  // class Conv2dToSparseConv2dMutator2

Expr Conv2dToSparse2(const Expr& e, const String& layout, int kernel_size, int blockH, int blockW,
                     double sparse_thresh) {
  auto rewriter = Conv2dToSparseConv2dMutator2(layout, kernel_size, blockH, blockW, sparse_thresh);
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

// Convert a model with separate weight info (already sparsified).
Pass Conv2dToSparse(const Array<ObjectRef>& weight_name, const Array<Array<PrimExpr>>& weight_shape,
                    const String& layout, int kernel_size) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // Remove FreeVar warnings
        auto f0 =
            Downcast<Function>(Conv2dToSparse(f, weight_name, weight_shape, layout, kernel_size));
        Array<Var> sparse_params = FreeVars(f0);
        auto f1 = WithFields(f0, sparse_params);
        Array<Var> params = FreeVars(f1);
        for (const auto& var : sparse_params) {
          params.push_back(var);
        }
        return WithFields(f1, params);
      };
  return CreateFunctionPass(pass_func, 4, "Conv2dToSparse", {"DeadCodeElimination"});
}

TVM_REGISTER_GLOBAL("relay._transform.Conv2dToSparse").set_body_typed(Conv2dToSparse);

// Convert a model with freezed params (sparsified in the pass).
Pass Conv2dToSparse2(const String& layout, int kernel_size, int blockH, int blockW,
                     double sparse_thresh) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto f0 = Downcast<Function>(
            Conv2dToSparse2(f, layout, kernel_size, blockH, blockW, sparse_thresh));
        return f0;
      };
  return CreateFunctionPass(pass_func, 5, "Conv2dToSparse2", {"DeadCodeElimination"});
}

TVM_REGISTER_GLOBAL("relay._transform.Conv2dToSparse2").set_body_typed(Conv2dToSparse2);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
