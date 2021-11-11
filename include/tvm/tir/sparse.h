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
 * \brief tvm/tir/sparse.h
 * \brief sparse axes and buffers.
 */
#ifndef TVM_TIR_SPARSE_H_
#define TVM_TIR_SPARSE_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace tir {

/*!
 * \brief Base type for axis in sparse formats.
 */
class AxisNode : public Object {
 public:
  /* name of current axis. */
  String name;
  /* length of current axis. For sparse axis, length refers to the upperbound of
   * the current axis. */
  PrimExpr length;

  String GetName() const { return name; }
  PrimExpr GetLength() const { return length; }
  DataType GetIndexType() const { return length->dtype; }

  static constexpr const char* _type_key = "tir.sparse.Axis";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(AxisNode, Object);
};

/*!
 * \brief Managed reference to AxisNode.
 * \sa AxisNode
 */
class Axis : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Axis, ObjectRef, AxisNode);
};

/*!
 * \brief Root of Axis Dependency Tree.
 */
class RootAxisNode : public Object {
 public:
  static constexpr const char* _type_key = "tir.sparse.RootAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(RootAxisNode, Object);
};

/*!
 * \brief Managed reference to RootAxisNode.
 * \sa RootAxisNode
 */
class RootAxis : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RootAxis, ObjectRef, RootAxisNode);
};

/*!
 * \brief Dense axis whose column indices are consecutive.
 */
class DenseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "tir.sparse.DenseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(DenseAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to DenseAxisNode.
 * \sa DenseAxisNode
 */
class DenseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DenseAxis, Axis, DenseAxisNode);
};

/*!
 * \brief Sparse axis whose column indices is not consecutive.
 */
class SparseAxisNode : public AxisNode {
 public:
  static constexpr const char* _type_key = "tir.sparse.SparseAxis";
  TVM_DECLARE_BASE_OBJECT_INFO(SparseAxisNode, AxisNode);
};

/*!
 * \brief Managed reference to SparseAxisNode.
 * \sa SparseAxisNode
 */
class SparseAxis : public Axis {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SparseAxis, Axis, SparseAxisNode);
};

/*!
 * \brief Dense axis with fixed length per row.
 */
class DenseFixedAxisNode : public DenseAxisNode {
 public:
  Optional<SparseAxis> from_sparse;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("length", &length);
    v->Visit("from_sparse", &from_sparse);
  }

  bool SEqualReduce(const DenseFixedAxisNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(name, other->name) && equal(length, other->length) &&
           equal(from_sparse, other->from_sparse);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(name);
    hash_reduce(length);
    hash_reduce(from_sparse);
  }

  static constexpr const char* _type_key = "tir.sparse.DenseFixedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DenseFixedAxisNode, DenseAxisNode);
};

/*!
 * \brief Managed reference to DenseFixedAxisNode.
 * \sa DenseFixedAxisNode
 */
class DenseFixedAxis : public DenseAxis {
 public:
  TVM_DLL explicit DenseFixedAxis(String name, PrimExpr length,
                                  Optional<SparseAxis> from_sparse = NullOpt);

  TVM_DEFINE_OBJECT_REF_METHODS(DenseFixedAxis, DenseAxis, DenseFixedAxisNode);
};

class DenseVariableAxisNode : public DenseAxisNode {
 public:
  Buffer indptr;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("length", &length);
    v->Visit("indptr", &indptr);
  }

  bool SEqualReduce(const DenseVariableAxisNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(name, other->name) && equal(length, other->length) && equal(indptr, other->indptr);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(name);
    hash_reduce(length);
    hash_reduce(indptr);
  }

  static constexpr const char* _type_key = "tir.sparse.DenseVariableAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(DenseVariableAxisNode, DenseAxisNode);
};

/*!
 * \brief Dense axis whose length is dependent on its predecessors on the axis
 * dependency tree.
 */
class DenseVariableAxis : public DenseAxis {
 public:
  TVM_DLL explicit DenseVariableAxis(String name, PrimExpr length, Buffer indptr);

  TVM_DEFINE_OBJECT_REF_METHODS(DenseVariableAxis, DenseAxis, DenseVariableAxisNode);
};

/*!
 * \brief Sparse axis with fixed number of non-zero columns per row.
 */
class SparseFixedAxisNode : public SparseAxisNode {
 public:
  Buffer indices;
  /* fixed number of columns of current sparse axis. */
  PrimExpr num_cols;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("length", &length);
    v->Visit("indptr", &indices);
    v->Visit("num_cols", &num_cols);
  }

  bool SEqualReduce(const SparseFixedAxisNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(name, other->name) && equal(length, other->length) &&
           equal(indices, other->indices) && equal(num_cols, other->num_cols);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(name);
    hash_reduce(length);
    hash_reduce(indices);
    hash_reduce(num_cols);
  }

  static constexpr const char* _type_key = "tir.sparse.SparseFixedAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseFixedAxisNode, SparseAxisNode);
};

/*!
 * \brief Managed reference to SparseFixedAxisNode.
 * \sa SparseFixedAxisNode
 */
class SparseFixedAxis : public SparseAxis {
 public:
  TVM_DLL explicit SparseFixedAxis(String name, PrimExpr length, Buffer indices, PrimExpr num_cols);

  TVM_DEFINE_OBJECT_REF_METHODS(SparseFixedAxis, SparseAxis, SparseFixedAxisNode);
};

/*!
 * \brief Sparse axis with variable number of non-zero columns per row.
 */
class SparseVariableAxisNode : public SparseAxisNode {
 public:
  Buffer indptr;
  Buffer indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("length", &length);
    v->Visit("indptr", &indptr);
    v->Visit("indices", &indices);
  }

  bool SEqualReduce(const SparseVariableAxisNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(name, other->name) && equal(length, other->length) &&
           equal(indptr, other->indptr) && equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(name);
    hash_reduce(length);
    hash_reduce(indptr);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "tir.sparse.SparseVariableAxis";
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseVariableAxisNode, SparseAxisNode);
};

/*!
 * \brief Managed reference to SparseVariableAxisNode.
 * \sa SparseVariableAxisNode
 */
class SparseVariableAxis : public SparseAxis {
 public:
  TVM_DLL explicit SparseVariableAxis(String name, PrimExpr length, Buffer indptr, Buffer indices);

  TVM_DEFINE_OBJECT_REF_METHODS(SparseVariableAxis, SparseAxis, SparseVariableAxisNode);
};

/*!
 * \brief Axis Dependency Tree.
 */
class AxisTreeNode : public Object {
 public:
  // unordered map that stores the parent relationship between axes.
  Map<String, Optional<String>> parent;
  // unordered map that stores the children relationship between axes.
  Map<Optional<String>, Array<String>> children;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("parent", &parent);
    v->Visit("children", &children);
  }

  bool SEqualReduce(const AxisTreeNode* other, SEqualReducer equal) const {
    return equal(parent, other->parent) && equal(children, other->children);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(parent);
    hash_reduce(children);
  }

  static constexpr const char* _type_key = "tir.sparse.AxisTree";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(AxisTreeNode, Object);
};

/*!
 * \brief Managed reference to AxisRefNode.
 * \sa AxisTreeNode
 */
class AxisTree : public ObjectRef {
 public:
  TVM_DLL AxisTree(Array<String> axis_names, Array<Optional<String>> axis_parent_names);

  TVM_DEFINE_OBJECT_REF_METHODS(AxisTree, ObjectRef, AxisTreeNode);
};

/*!
 * \brief Class of sparse buffer.
 */
class SparseBufferNode : public Object {
 public:
  /* Axes */
  Array<Axis> axes;
  /* Buffer corresponding to flattened value */
  Buffer data;
  /* Buffer Name */
  String name;

  inline int ndim() const { return static_cast<int>(axes.size()); }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("axes", &axes);
    v->Visit("data", &data);
    v->Visit("name", &name);
  }

  bool SEqualReduce(const SparseBufferNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(axes, other->axes) && equal(data, other->data) && equal(name, other->name);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(axes);
    hash_reduce(data);
    hash_reduce(name);
  }

  static constexpr const char* _type_key = "tir.sparse.SparseBuffer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SparseBufferNode, Object);
};

/*!
 * \brief Managed reference to SparseBufferNode.
 * \sa SparseBufferNode
 */
class SparseBuffer : public ObjectRef {
 public:
  TVM_DLL explicit SparseBuffer(Array<Axis> axes, Buffer data, String name);

  TVM_DEFINE_OBJECT_REF_METHODS(SparseBuffer, ObjectRef, SparseBufferNode);
};

enum class SpIterKind : int {
  kDenseFixed = 0,
  kDenseVariable = 1,
  kSparseFixed = 2,
  kSparseVariable = 3
};

// overload printing of for type.
TVM_DLL std::ostream& operator<<(std::ostream& os, SpIterKind kind);

/*!
 * \brief Iterator variables in SparseTIR
 */
class SpIterVarNode : public Object {
 public:
  Var var;
  PrimExpr max_extent;
  SpIterKind kind;
  bool is_reduction;
  Axis axis;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("max_extent", &max_extent);
    v->Visit("axis", &axis);
    v->Visit("is_reduction", &is_reduction);
    v->Visit("kind", &kind);
  }

  bool SEqualReduce(const SpIterVarNode* other, SEqualReducer equal) const {
    return equal(var, other->var) && equal(max_extent, other->max_extent) &&
           equal(axis, other->axis) && equal(is_reduction, other->is_reduction) &&
           equal(kind, other->kind);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(var);
    hash_reduce(max_extent);
    hash_reduce(axis);
    hash_reduce(is_reduction);
    hash_reduce(kind);
  }

  static constexpr const char* _type_key = "tir.sparse.SpIterVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SpIterVarNode, Object);
};

class SpIterVar : public ObjectRef {
 public:
  TVM_DLL explicit SpIterVar(Var var, PrimExpr max_extent, SpIterKind kind, bool is_reduction,
                             Axis axis);

  /*!
   * \return the corresponding var in the IterVar.
   */
  inline operator PrimExpr() const;

  TVM_DEFINE_OBJECT_REF_METHODS(SpIterVar, ObjectRef, SpIterVarNode);
};

// inline implementations
inline SpIterVar::operator PrimExpr() const { return (*this)->var; }

// inline implementations
inline const char* SpIterKind2String(SpIterKind t) {
  switch (t) {
    case SpIterKind::kDenseFixed:
      return "dense_fixed";
    case SpIterKind::kDenseVariable:
      return "dense_variable";
    case SpIterKind::kSparseFixed:
      return "sparse_fixed";
    case SpIterKind::kSparseVariable:
      return "sparse_variable";
  }
  LOG(FATAL) << "Unknown SpIterKind" << t;
  throw;
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SPARSE_H_
