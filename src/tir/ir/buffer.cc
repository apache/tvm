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
 * \file buffer.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include "buffer_common.h"

#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

Array<PrimExpr> SimplifyArray(arith::Analyzer* ana, Array<PrimExpr> array) {
  for (size_t i = 0; i < array.size(); ++i) {
    array.Set(i, ana->Simplify(array[i]));
  }
  return array;
}

Buffer decl_buffer(Array<PrimExpr> shape, DataType dtype, String name, String storage_scope,
                   Span span) {
  DataType storage_dtype = (dtype == DataType::Bool() ? DataType::Int(8) : dtype);
  return Buffer(Var(name, PointerType(PrimType(storage_dtype), storage_scope), span), dtype, shape,
                Array<PrimExpr>(), PrimExpr(), name, 0, 0, kDefault, span);
}

PrimExpr Buffer::vload(Array<PrimExpr> begin, DataType dtype) const {
  // specially handle bool, stored as DataType::Int(8)
  const BufferNode* n = operator->();
  ICHECK(dtype.element_of() == n->dtype.element_of() && dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot load " << dtype << " from buffer of " << n->dtype;
  if (dtype == DataType::Bool()) {
    return tir::Cast(DataType::Bool(),
                     tir::Load(DataType::Int(8), n->data, BufferOffset(n, begin, DataType::Int(8)),
                               const_true()));
  } else {
    return tir::Load(dtype, n->data, BufferOffset(n, begin, dtype), const_true(dtype.lanes()));
  }
}

Stmt Buffer::vstore(Array<PrimExpr> begin, PrimExpr value) const {
  // specially handle bool, stored as DataType::Int(8)
  const BufferNode* n = operator->();
  DataType dtype = value.dtype();
  ICHECK(dtype.element_of() == n->dtype.element_of() && dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot store " << dtype << " to buffer of " << n->dtype;
  if (value.dtype() == DataType::Bool()) {
    return tir::Store(n->data, tir::Cast(DataType::Int(8), value),
                      BufferOffset(n, begin, DataType::Int(8)), const_true());
  } else {
    return tir::Store(n->data, value, BufferOffset(n, begin, dtype), const_true(dtype.lanes()));
  }
}

String Buffer::scope() const {
  const auto* ptr_type = (*this)->data->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type) << "Buffer variable is not of pointer type";
  if (ptr_type->storage_scope.empty()) {
    return "global";
  }
  return ptr_type->storage_scope;
}

Buffer Buffer::MakeStrideView() const {
  if ((*this)->strides.size() != 0) return *this;
  if ((*this)->shape.size() == 0) return *this;
  std::vector<PrimExpr> temp;
  auto n = make_object<BufferNode>(*operator->());
  PrimExpr acc = make_const(n->DefaultIndexType(), 1);
  for (size_t i = n->shape.size(); i != 0; --i) {
    temp.push_back(acc);
    acc = acc * n->shape[i - 1];
  }
  for (size_t i = temp.size(); i != 0; --i) {
    n->strides.push_back(temp[i - 1]);
  }
  return Buffer(n);
}

Buffer Buffer::MakeSlice(Array<PrimExpr> begins, Array<PrimExpr> extents) const {
  const BufferNode* n = operator->();
  arith::Analyzer ana;
  begins = SimplifyArray(&ana, begins);
  PrimExpr elem_offset = ana.Simplify(ElemOffset(n, begins));
  Array<PrimExpr> strides = n->strides;
  if (strides.size() == 0) {
    bool can_relax = true;
    bool need_stride = false;
    // check if stride is needed.
    for (size_t i = 0; i < extents.size(); ++i) {
      if (!can_relax) {
        if (!is_zero(begins[i]) || !is_zero(ana.Simplify(extents[i] - n->shape[i]))) {
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
  return Buffer(n->data, n->dtype, extents, strides, elem_offset, n->name + "_slice",
                n->data_alignment, 0, n->buffer_type);
}

PrimExpr Buffer::access_ptr(int access_mask, DataType ptr_type, int content_lanes,
                            PrimExpr offset) const {
  const BufferNode* self = operator->();
  PrimExpr e_dtype;
  PrimExpr extent;
  if (self->shape.size() == 0) {
    extent = make_const(self->DefaultIndexType(), 1);
  } else if (self->strides.size() == self->shape.size()) {
    int highest_dim = 0;
    extent = self->strides[highest_dim] * self->shape[highest_dim] - offset;
  } else {
    extent = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                   make_const(DataType::Int(32), 1), self->shape) -
             offset;
  }
  PrimExpr elem_offset = self->elem_offset + offset;
  if (content_lanes > 1) {
    e_dtype = tir::TypeAnnotation(self->dtype.with_lanes(content_lanes));
    extent = extent / make_const(self->elem_offset.dtype(), content_lanes);
    elem_offset = self->elem_offset / make_const(self->elem_offset.dtype(), content_lanes);
  } else {
    e_dtype = tir::TypeAnnotation(self->dtype);
  }
  Array<PrimExpr> acc_args{e_dtype, self->data, elem_offset, extent,
                           make_const(DataType::Int(32), access_mask)};
  return tir::Call(ptr_type, tir::builtin::tvm_access_ptr(), acc_args);
}

Buffer::Buffer(Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
               PrimExpr elem_offset, String name, int data_alignment, int offset_factor,
               BufferType buffer_type, Span span) {
  DataType storage_dtype = dtype;
  // specially handle bool
  if (storage_dtype == DataType::Bool()) {
    storage_dtype = DataType::Int(8);
  }
  ICHECK(IsPointerType(data->type_annotation, storage_dtype) ||
         IsTextureType(data->type_annotation, storage_dtype))
      << "Buffer data field expect to have the right pointer type annotation"
      << " annotation=" << data->type_annotation << ", storage_dtype=" << storage_dtype;

  auto n = make_object<BufferNode>();
  n->data = std::move(data);
  n->dtype = dtype;

  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->name = std::move(name);
  if (!elem_offset.defined()) {
    elem_offset = make_const(n->DefaultIndexType(), 0);
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
  n->buffer_type = buffer_type;
  if (n->buffer_type == kAutoBroadcast && n->shape.size() > 0 && n->strides.empty()) {
    for (size_t i = 0; i < n->shape.size(); ++i) {
      n->strides.push_back(Var("stride", n->shape[i].dtype()));
    }
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferNode*>(node.get());
      p->stream << "buffer(" << op->name << ", " << op << ")";
    });

TVM_REGISTER_NODE_TYPE(BufferNode);

TVM_REGISTER_GLOBAL("tir.Buffer").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args.size(), 10);
  auto buffer_type = args[8].operator String();
  BufferType type = (buffer_type == "auto_broadcast") ? kAutoBroadcast : kDefault;
  *ret =
      Buffer(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], type, args[9]);
});

TVM_REGISTER_GLOBAL("tir.BufferAccessPtr").set_body_method(&Buffer::access_ptr);

TVM_REGISTER_GLOBAL("tir.BufferVLoad").set_body_method(&Buffer::vload);

TVM_REGISTER_GLOBAL("tir.BufferVStore").set_body_method(&Buffer::vstore);

TVM_REGISTER_GLOBAL("tir.BufferStorageScope").set_body_method(&Buffer::scope);

}  // namespace tir
}  // namespace tvm
