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
 * \brief Scan Operator.
 * \file scan_op.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace te {
using namespace tir;

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ScanOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ScanOpNode*>(node.get());
      p->stream << "scan(" << op->name << ", " << op << ")";
    });
TVM_REGISTER_NODE_TYPE(ScanOpNode);

int ScanOpNode::num_outputs() const { return static_cast<int>(update.size()); }

DataType ScanOpNode::output_dtype(size_t i) const { return update[i]->dtype; }

Array<PrimExpr> ScanOpNode::output_shape(size_t i) const {
  ICHECK_LT(i, state_placeholder.size());
  return state_placeholder[i]->shape;
}

ScanOp::ScanOp(std::string name, std::string tag, Optional<Map<String, ffi::Any>> attrs,
               IterVar axis, Array<Tensor> init, Array<Tensor> update,
               Array<Tensor> state_placeholder, Array<Tensor> inputs) {
  if (!attrs.defined()) {
    attrs = Map<String, ffi::Any>();
  }
  auto n = make_object<ScanOpNode>();
  ICHECK_EQ(init.size(), update.size());
  ICHECK_EQ(init.size(), state_placeholder.size());
  arith::Analyzer analyzer;
  auto prove_equal = [&](PrimExpr lhs, PrimExpr rhs) {
    return is_zero(analyzer.Simplify(lhs - rhs));
  };

  for (size_t i = 0; i < init.size(); ++i) {
    ICHECK_EQ(init[i]->dtype, state_placeholder[i]->dtype);
    ICHECK_EQ(init[i]->dtype, update[i]->dtype);
    ICHECK(prove_equal(init[i]->shape[0], axis->dom->min))
        << "init.shape[0] need to match scan_axis.dom.min";
    ICHECK(prove_equal(state_placeholder[i]->shape[0], axis->dom->min + axis->dom->extent))
        << "state_placeholder.shape[0] need to match"
        << " scan_axis.dom.min + scan_axis.dom.extent";
    ICHECK_EQ(state_placeholder[i].ndim(), init[i].ndim())
        << "The dimension of init need to match state_placeholder";
    ICHECK_EQ(update[i].ndim(), state_placeholder[i].ndim())
        << "The update.ndim need to be state_placeholder.ndim - 1";
    for (size_t k = 0; k < update[i].ndim(); ++k) {
      ICHECK(prove_equal(update[i]->shape[k], state_placeholder[i]->shape[k]));
      if (k != 0) {
        // setup spatial axis
        std::ostringstream spatial_name;
        spatial_name << name << ".out" << i << ".i" << k;
        n->spatial_axis_.push_back(IterVar(Range::FromMinExtent(0, update[i]->shape[k]),
                                           Var(spatial_name.str()), kOpaque));
      }
    }

    for (size_t k = 1; k < init[i].ndim(); ++k) {
      ICHECK(prove_equal(init[i]->shape[k], state_placeholder[i]->shape[k]));
    }
  }
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = attrs.value_or({});
  n->scan_axis = std::move(axis);
  n->init = std::move(init);
  n->update = std::move(update);
  n->state_placeholder = std::move(state_placeholder);
  n->inputs = std::move(inputs);
  data_ = std::move(n);
}

TVM_FFI_REGISTER_GLOBAL("te.ScanOp")
    .set_body_typed([](std::string name, std::string tag, Optional<Map<String, ffi::Any>> attrs,
                       IterVar axis, Array<Tensor> init, Array<Tensor> update,
                       Array<Tensor> state_placeholder, Array<Tensor> inputs) {
      return ScanOp(name, tag, attrs, axis, init, update, state_placeholder, inputs);
    });

Array<Tensor> scan(Array<Tensor> init, Array<Tensor> update, Array<Tensor> state_placeholder,
                   Array<Tensor> inputs, std::string name, std::string tag,
                   Optional<Map<String, ffi::Any>> attrs) {
  IterVar scan_axis =
      IterVar(Range::FromMinExtent(init[0]->shape[0], update[0]->shape[0] - init[0]->shape[0]),
              Var(name + ".idx"), kOrdered);
  Operation op = ScanOp(name, tag, attrs, scan_axis, init, update, state_placeholder, inputs);
  Array<Tensor> res;
  for (int i = 0; i < op->num_outputs(); ++i) {
    res.push_back(op.output(i));
  }
  return res;
}

Array<Tensor> ScanOpNode::InputTensors() const {
  Array<Tensor> ret;
  for (Tensor t : init) {
    ret.push_back(t);
  }
  for (Tensor t : update) {
    ret.push_back(t);
  }
  return ret;
}

}  // namespace te
}  // namespace tvm
