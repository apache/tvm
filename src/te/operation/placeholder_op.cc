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
 * \brief Placeholder op.
 * \file placeholder_op.cc
 */
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/te/operation.h>

namespace tvm {
namespace te {

TVM_FFI_STATIC_INIT_BLOCK({ PlaceholderOpNode::RegisterReflection(); });

// PlaceholderOpNode
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PlaceholderOpNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PlaceholderOpNode*>(node.get());
      p->stream << "placeholder(" << op->name << ", " << op << ")";
    });

int PlaceholderOpNode::num_outputs() const { return 1; }

DataType PlaceholderOpNode::output_dtype(size_t i) const {
  ICHECK_EQ(i, 0U);
  return dtype;
}

ffi::Array<PrimExpr> PlaceholderOpNode::output_shape(size_t i) const {
  ICHECK_EQ(i, 0U);
  return shape;
}

PlaceholderOp::PlaceholderOp(std::string name, ffi::Array<PrimExpr> shape, DataType dtype) {
  auto n = ffi::make_object<PlaceholderOpNode>();
  n->name = name;
  n->shape = shape;
  n->dtype = dtype;
  data_ = std::move(n);
}

Tensor placeholder(ffi::Array<PrimExpr> shape, DataType dtype, std::string name) {
  return PlaceholderOp(name, shape, dtype).output(0);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("te.Placeholder", [](ffi::Variant<PrimExpr, ffi::Array<PrimExpr>> shape_arg,
                                             DataType dtype, std::string name) {
    auto shape = [&]() -> ffi::Array<PrimExpr> {
      if (auto arg_expr = shape_arg.as<PrimExpr>()) {
        return {arg_expr.value()};
      } else if (auto arg_array = shape_arg.as<ffi::Array<PrimExpr>>()) {
        return arg_array.value();
      } else {
        LOG(FATAL) << "Variant did not contain either allowed type";
      }
    }();
    return placeholder(shape, dtype, name);
  });
});

ffi::Array<Tensor> PlaceholderOpNode::InputTensors() const { return {}; }

}  // namespace te
}  // namespace tvm
