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
 * \file tvm/tir/buffer.h
 * \brief Symbolic n-dimensional array, to represent a memory buffer.
 */
#ifndef TVM_TIR_BUFFER_H_
#define TVM_TIR_BUFFER_H_

#include <tvm/ir/expr.h>
#include <tvm/node/script_printer.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/var.h>

#include <string>

namespace tvm {
namespace tir {

#ifndef TVM_INDEX_DEFAULT_I64
#define TVM_INDEX_DEFAULT_I64 1
#endif
/*! \brief if TVM_INDEX_DEFAULT_I64 is set, return int64, otherwise return int32 */
inline DataType DefaultIndexType() {
#if TVM_INDEX_DEFAULT_I64
  return DataType::Int(64);
#else
  return DataType::Int(32);
#endif
}

// forward declare Stmt
class Stmt;

/*! \brief buffer type */
enum BufferType : int {
  kDefault = 1,
  // Maps buffer[i][j][k] -> buffer[i][0][k] if dimension i's shape equals 1.
  kAutoBroadcast = 2,
};

/*! \brief Node to represent a buffer */
class BufferNode : public Object {
 public:
  // Data fields.
  /*!
   * \brief The pointer to the head of the data
   * \sa data_alignment The alignment of data in bytes.
   */
  Var data;
  /*! \brief data type in the content of the tensor */
  DataType dtype;
  /*! \brief The type of the buffer prior to flattening
   *
   * This contains the shape as it is accessed by
   * BufferLoad/BufferStore nodes, and used by the low-level code
   * generators.
   */
  Array<PrimExpr> shape;
  /*!
   * \brief Separators between input axes when generating flattened output axes
   *
   * For buffers representing flat 1-d memory (e.g. any buffer in
   * RAM), this should be an empty array.  For buffers representing
   * non-flat memory, each entry in axis_separators should be the
   * first input axis that is part of a new flattened axis.
   */
  Array<IntImm> axis_separators;
  /*!
   * \brief The strides of each dimension
   *  This can be an empty array, indicating array is contiguous
   */
  Array<PrimExpr> strides;
  /*! \brief The offset in terms of number of dtype elements (including lanes) */
  PrimExpr elem_offset;
  // Meta data
  /*! \brief optional name of the buffer */
  String name;
  /*! \brief Alignment requirement of data pointer in bytes. */
  int data_alignment;
  /*!
   * \brief Factor of elem_offset field,
   *  elem_offset is guaranteed to be multiple of offset_factor.
   */
  int offset_factor;
  /*! \brief buffer type */
  BufferType buffer_type;
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;
  /*! \brief constructor */
  BufferNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
    v->Visit("strides", &strides);
    v->Visit("axis_separators", &axis_separators);
    v->Visit("elem_offset", &elem_offset);
    v->Visit("name", &name);
    v->Visit("data_alignment", &data_alignment);
    v->Visit("offset_factor", &offset_factor);
    v->Visit("buffer_type", &buffer_type);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const BufferNode* other, SEqualReducer equal) const {
    // Use DefEqual as buffer can define variables in its semantics,
    // skip name as name is not important.
    return equal.DefEqual(data, other->data) && equal(dtype, other->dtype) &&
           equal.DefEqual(shape, other->shape) && equal.DefEqual(strides, other->strides) &&
           equal.DefEqual(axis_separators, other->axis_separators) &&
           equal.DefEqual(elem_offset, other->elem_offset) &&
           equal(data_alignment, other->data_alignment) && equal(buffer_type, other->buffer_type);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(data);
    hash_reduce(dtype);
    hash_reduce.DefHash(shape);
    hash_reduce.DefHash(strides);
    hash_reduce.DefHash(elem_offset);
    hash_reduce.DefHash(axis_separators);
    hash_reduce(data_alignment);
    hash_reduce(buffer_type);
  }

  /*! \return preferred index type for this buffer node */
  DataType DefaultIndexType() const {
    return shape.size() != 0 ? shape[0].dtype() : tvm::tir::DefaultIndexType();
  }

  /*! \brief Determine the offset in the buffer of the given index.
   *
   * Returns the buffer offset, in number of elements of type dtype,
   * without adjusting for number of lanes.  (e.g. The number of
   * float16x4 elements in a buffer of type float16x4.)
   */
  Array<PrimExpr> ElemOffset(Array<PrimExpr> index) const;

  static constexpr const char* _type_key = "tir.Buffer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferNode, Object);
  TVM_OBJECT_ENABLE_SCRIPT_PRINTER();
};

/*!
 * \brief Buffer is a symbolic n-darray structure.
 *  It is a composition of primitive symbolic types,
 *  used to specify the memory layout of the Tensor used in program input.
 */
class Buffer : public ObjectRef {
 public:
  // User can specify data_alignment and offset_factor to be 0
  // A default value will be picked.
  TVM_DLL Buffer(Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
                 PrimExpr elem_offset, String name, int data_alignment, int offset_factor,
                 BufferType buffer_type, Array<IntImm> axis_separators = {}, Span span = Span());

  /*!
   * \brief Return a new buffer that is equivalent with current one
   *  but always add stride field.
   * \return The strided version of the buffer.
   */
  TVM_DLL Buffer MakeStrideView() const;
  /*!
   * \brief Make a new symbolic buffer representing a slice of the buffer.
   * \param begins The beginning position of each dimension.
   * \param extents The extent of each dimension.
   * \note This function will make target buffer as compact as possible.
   *  If stride is not needed in the slice, it won't be presented
   * \return the result buffer.
   */
  TVM_DLL Buffer MakeSlice(Array<PrimExpr> begins, Array<PrimExpr> extents) const;
  /*!
   * \brief Get access ptr to the entire buffer.
   * \param access_mask The access mask
   * \param ptr_type The type of the pointer.
   * \param content_lanes The number of lanes for the (data) type.
   * \param offset The offset of ptr.
   * \param input_extent The extent of ptr.
   */
  TVM_DLL PrimExpr access_ptr(int access_mask, DataType ptr_type = DataType::Handle(),
                              int content_lanes = 1, PrimExpr offset = IntImm(DataType::Int(32), 0),
                              Optional<PrimExpr> input_extent = NullOpt) const;
  /*!
   * \brief Create an Expr that does a vector load at begin index.
   * \param begin The beginning index
   * \param dtype The data type to be loaded.
   */
  TVM_DLL PrimExpr vload(Array<PrimExpr> begin, DataType dtype) const;
  /*!
   * \brief Create a Stmt that does a vector store at begin index.
   * \param begin The beginning index
   * \param value The value to be stored.
   */
  TVM_DLL Stmt vstore(Array<PrimExpr> begin, PrimExpr value) const;

  /*!
   * \brief Get a flattened version of the buffer
   */
  Buffer GetFlattenedBuffer() const;

  /*! \brief Determine the offset in the buffer of the given index.
   *
   * Returns the buffer offset, in number of elements of type dtype,
   * without adjusting for number of lanes.  (e.g. The number of
   * float16x4 elements in a buffer of type float16x4.)
   */
  Array<PrimExpr> OffsetOf(Array<PrimExpr> index) const;

  /*!
   * \brief Return the storage scope associated with this buffer.
   */
  TVM_DLL String scope() const;

  TVM_DEFINE_OBJECT_REF_METHODS(Buffer, ObjectRef, BufferNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BufferNode);
};

/*!
 * \brief Construct a new buffer given shape, and dtype.
 * \param shape The shape of the buffer,
 * \param dtype The content data type.
 * \param name The name of the buffer
 * \param storage_scope The storage scope associated with this buffer
 * \param axis_separators Divisions defining the groups of axes that will be flattened together.
 * \param span The location of this object in the source code.
 * \return The created buffer.
 * \sa Buffer for complete constructor.
 */
TVM_DLL Buffer decl_buffer(Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                           String name = "buffer", String storage_scope = "",
                           Array<IntImm> axis_separators = {}, Span span = Span());

/*!
 * \brief Base node for data producers.
 *
 *  A DataProducer stores necessary information(e.g. a tensor expression) to produce
 *  a multi-dimensional array. The stored information is opaque to the TIR.
 *  DataProducer can appear in high-level DSLs that are built on top of the TIR.
 *
 *  A valid TIR PrimFunc should not contain any DataProducer, high level DSLs should lower
 *  all DataProducers to Buffers before TIR transformations.
 *
 * \sa tvm::te::Tensor
 */
class DataProducerNode : public Object {
 public:
  /*! \brief destructor. */
  virtual ~DataProducerNode() {}
  /*!
   * \brief Get the shape of the result.
   * \return The shape.
   */
  virtual Array<PrimExpr> GetShape() const = 0;
  /*!
   * \brief Get the data type of the result.
   * \return The data type.
   */
  virtual DataType GetDataType() const = 0;
  /*!
   * \brief Get the name hint of the data producer.
   * \return The data type.
   */
  virtual String GetNameHint() const = 0;

  bool SEqualReduce(const DataProducerNode* other, SEqualReducer equal) const {
    // because buffer producer is opaque, we just do pointer equality.
    return this == other;
  }

  void SHashReduce(SHashReducer hash_reduce) const {}

  static constexpr const char* _type_key = "tir.DataProducer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(DataProducerNode, Object);
};

/*!
 * \brief Managed reference to DataProducerNode.
 * \sa DataProducerNode
 */
class DataProducer : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(DataProducer, ObjectRef, DataProducerNode);
};

/*!
 * \brief Creates TIR Buffer for provided parameters
 * \param shape shape of the buffer
 * \param dtype data type
 * \param name buffer name
 * \param data_alignment alignment requirement of data pointer in bytes
 * \param offset_factor Factor of elem_offset field, elem_offset is guaranteed to be
 *                      multiple of offset_factor
                        User can specify data_alignment and offset_factor to be 0
 *                      A default value will be picked.
 * \param compact If the statement has already bound to a compact buffer.
 * \param memory_scope memory scope of the buffer
 */
TVM_DLL tir::Buffer BufferWithOffsetAlignment(Array<PrimExpr> shape, DataType dtype,
                                              std::string name, int data_alignment,
                                              int offset_factor, bool compact,
                                              std::string memory_scope = "");
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_BUFFER_H_
