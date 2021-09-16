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
 * \file relay/backend/contrib/ethosu/to_te_graph.cc
 * \brief Lower a Relay function to a TE graph.
 */
#include <tvm/driver/driver_api.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/tags.h>

#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../compile_engine.h"
#include "../../utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace ethosu {

/*! \brief Node container to represent a Tensor Expression graph. */
class TEGraphNode : public Object {
 public:
  /* \brief The inputs to the graph */
  tvm::Array<te::Tensor> inputs;
  /* \brief The outputs to the graph */
  tvm::Array<te::Tensor> outputs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
  }

  static constexpr const char* _type_key = "relay.TEGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(TEGraphNode, Object);
};

class TEGraph : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(TEGraph, ObjectRef, TEGraphNode);
};

TVM_REGISTER_NODE_TYPE(TEGraphNode);

Array<IndexExpr> GetShape(const Array<IndexExpr>& shape) {
  // for now, we always use int32 shape when possible
  // even if the result of shape inference becomes int64.
  Array<IndexExpr> res;
  for (IndexExpr val : shape) {
    const int64_t* pval = tir::as_const_int(val);
    if (pval != nullptr) {
#ifndef TVM_INDEX_DEFAULT_I64
      ICHECK_LE(pval[0], std::numeric_limits<int32_t>::max());
      ICHECK_GE(pval[0], std::numeric_limits<int32_t>::min());
      res.push_back(IntImm(DataType::Int(32), *pval));
#else
      res.push_back(val);
#endif  // TVM_INDEX_DEFAULT_I64
    } else if (val->IsInstance<tir::AnyNode>()) {
      res.push_back(val.as<tir::AnyNode>()->ToVar());
    } else {
      res.push_back(val);
    }
  }
  return res;
}

class RelayToTE : public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  RelayToTE() = default;

  TEGraph Lower(const Function& prim_func) {
    auto graph_node = make_object<TEGraphNode>();
    for (Var param : prim_func->params) {
      Array<tvm::te::Tensor> inputs;
      if (const auto* ttype = param->checked_type().as<TensorTypeNode>()) {
        tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype);
        graph_node->inputs.push_back(tensor);
        inputs.push_back(tensor);
      } else {
        // flatten tuple of tensor type.
        const auto* tuple_type = param->type_as<TupleTypeNode>();
        for (Type field : tuple_type->fields) {
          const auto* ttype = field.as<TensorTypeNode>();
          ICHECK(ttype != nullptr);
          tvm::te::Tensor tensor = tvm::te::placeholder(GetShape(ttype->shape), ttype->dtype);
          graph_node->inputs.push_back(tensor);
          inputs.push_back(tensor);
        }
      }
      memo_[param] = inputs;
    }
    graph_node->outputs = this->VisitExpr(prim_func->body);
    return TEGraph(graph_node);
  }

  Array<te::Tensor> VisitExpr_(const VarNode* op) final {
    LOG(FATAL) << "Free variable " << op->name_hint();
    return {};
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    ICHECK(op->is_scalar());
    void* data = op->data->data;
    DataType dtype = DataType(op->data->dtype);
    auto value = te::compute(
        {},
        [&](const Array<tvm::tir::Var>&) {
          if (dtype == DataType::Int(32)) {
            return make_const(dtype, static_cast<const int32_t*>(data)[0]);
          } else if (dtype == DataType::Int(64)) {
            return make_const(dtype, static_cast<const int64_t*>(data)[0]);
          } else if (dtype == DataType::Float(32)) {
            return make_const(dtype, static_cast<const float*>(data)[0]);
          } else if (dtype == DataType::Float(64)) {
            return make_const(dtype, static_cast<const double*>(data)[0]);
          } else if (dtype == DataType::Bool()) {
            return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
          } else {
            LOG(FATAL) << "not handled";
            return tvm::PrimExpr();
          }
        },
        "compile_engine_const", topi::kBroadcast);
    return {value};
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    static auto flower_call = tvm::runtime::Registry::Get("relay.backend.lower_call");
    ICHECK(flower_call) << "relay.backend.lower_call is not registered.";

    Array<te::Tensor> inputs;
    int count_tuple = 0;
    for (Expr arg : call_node->args) {
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
    }
    if (count_tuple) {
      ICHECK_EQ(call_node->args.size(), 1U) << "Only allow function with a single tuple input";
    }

    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);

    Array<te::Tensor> outputs;
    LoweredOutput lowered_out =
        (*flower_call)(GetRef<Call>(call_node), inputs, tvm::Target("llvm"));
    outputs = lowered_out->outputs;

    if (outputs.size() != 1) {
      const auto* tuple_type = call_node->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type) << "Expect output to be a tuple type";
      ICHECK_EQ(tuple_type->fields.size(), outputs.size());
    }
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Do not support sub function";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      ICHECK(field->checked_type().as<TensorTypeNode>()) << "Only allow Tuple of Tensor";
      Array<te::Tensor> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    const auto* tuple_type = op->tuple->type_as<TupleTypeNode>();
    Array<te::Tensor> tuple = VisitExpr(op->tuple);
    ICHECK_EQ(tuple_type->fields.size(), tuple.size());
    ICHECK_GE(op->index, 0);
    ICHECK_LT(static_cast<size_t>(op->index), tuple.size());
    return {tuple[op->index]};
  }
};

TVM_REGISTER_GLOBAL("relay.backend.contrib.ethosu.LowerToTE")
    .set_body_typed([](Function prim_func) { return RelayToTE().Lower(prim_func); });

}  // namespace ethosu
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
