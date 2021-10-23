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
 * \file sparse.cc
 * \brief buffers and formats in sparse tir.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/sparse.h>

namespace tvm {
namespace tir {

// DenseFixedAxis
DenseFixedAxis::DenseFixedAxis(String name, PrimExpr length) {
  ObjectPtr<DenseFixedAxisNode> node = make_object<DenseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(DenseFixedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.DenseFixedAxis")
    .set_body_typed([](String name, PrimExpr length) {
      return DenseFixedAxis(name, length);
    });

// DenseVariableAxis
DenseVariableAxis::DenseVariableAxis(String name, PrimExpr length,
                                     Buffer indptr) {
  ObjectPtr<DenseVariableAxisNode> node = make_object<DenseVariableAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indptr = std::move(indptr);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(DenseVariableAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.DenseVariableAxis")
    .set_body_typed([](String name, PrimExpr length, Buffer indptr) {
      return DenseVariableAxis(name, length, indptr);
    });

// SparseFixedAxis
SparseFixedAxis::SparseFixedAxis(String name, PrimExpr length, Buffer indices,
                                 PrimExpr num_cols) {
  ObjectPtr<SparseFixedAxisNode> node = make_object<SparseFixedAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indices = std::move(indices);
  node->num_cols = std::move(num_cols);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseFixedAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseFixedAxis")
    .set_body_typed([](String name, PrimExpr length, Buffer indices,
                       PrimExpr num_cols) {
      return SparseFixedAxis(name, length, indices, num_cols);
    });

// SparseVariableAxis
SparseVariableAxis::SparseVariableAxis(String name, PrimExpr length,
                                       Buffer indptr, Buffer indices) {
  ObjectPtr<SparseVariableAxisNode> node =
      make_object<SparseVariableAxisNode>();
  node->name = std::move(name);
  node->length = std::move(length);
  node->indptr = std::move(indptr);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseVariableAxisNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseVariableAxis")
    .set_body_typed([](String name, PrimExpr length, Buffer indptr,
                       Buffer indices) {
      return SparseVariableAxis(name, length, indptr, indices);
    });

// AxisTree
AxisTree::AxisTree(Array<Axis> axes,
                   Array<Optional<String>> axis_parent_names) {
  CHECK_EQ(axes.size(), axis_parent_names.size())
      << "ValueError: The axes array should have the same length as axis_parent_names "
         "array.";
  ObjectPtr<AxisTreeNode> node = make_object<AxisTreeNode>();
  Axis root = Downcast<Axis>(RootAxis());
  for (const Axis& axis : axes) {
    // update axis map
    String name = axis->name;
    CHECK(node->axis_map.find(name) != node->axis_map.end()) << "ValueError: duplicate axis names.";
    node->axis_map[name] = axis;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    // update parent map & children map
    Axis axis = axes[i];
    Optional<String> parent_name = axis_parent_names[i];
    if (parent_name.get() != nullptr) {
      CHECK(node->axis_map.find(parent_name.value()) != node->axis_map.end())
          << "ValueError: Parent axis name doesn't exist.";
    }
    Axis parent_axis = (parent_name.get() != nullptr)
                           ? node->axis_map[parent_name.value()]
                           : root;
    node->parent[axis] = parent_axis;
    if (node->children.find(parent_axis) != node->children.end()) {
      node->children[parent_axis].push_back(axis);
    } else {
      Array<Axis> children;
      children.push_back(axis);
      node->children[parent_axis] = std::move(children);
    }
  }
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AxisTreeNode);

TVM_REGISTER_GLOBAL("tir.sparse.AxisTree")
    .set_body_typed([](Array<Axis> axes,
                       Array<Optional<String>> axis_parent_names) {
      return AxisTree(axes, axis_parent_names);
    });

// SparseBuffer
SparseBuffer::SparseBuffer(AxisTree tree, Array<Axis> axes, Buffer data, String name,
                           DataType dtype) {
  ObjectPtr<SparseBufferNode> node = make_object<SparseBufferNode>();
  node->tree = std::move(tree);
  node->axes = std::move(axes);
  node->data = std::move(data);
  node->name = std::move(name);
  node->dtype = dtype;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(SparseBufferNode);

TVM_REGISTER_GLOBAL("tir.sparse.SparseBuffer")
    .set_body_typed([](AxisTree tree, Array<Axis> axes, Buffer data, String name, DataType dtype) {
      return SparseBuffer(tree, axes, data, name, dtype);
    });

}  // namespace tir
}  // namespace tvm
