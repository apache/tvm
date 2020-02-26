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

#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <string>


namespace tvm {
namespace tir {
// Internal node container Buffer
class BufferNode;

/*! \brief buffer type */
enum BufferType : int {
  kDefault = 1,
  // Maps buffer[i][j][k] -> buffer[i][0][k] if dimension i's shape equals 1.
  kAutoBroadcast = 2,
};

/*!
 * \brief Buffer is a symbolic n-darray structure.
 *  It is a composition of primitive symbolic types,
 *  used to specify the memory layout of the Tensor used in program input.
 */
class Buffer : public ObjectRef {
 public:
  Buffer() {}
  explicit Buffer(ObjectPtr<Object> n) : ObjectRef(n) {}
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
   */
  TVM_DLL PrimExpr access_ptr(int access_mask,
                          DataType ptr_type = DataType::Handle(),
                          int content_lanes = 1,
                          PrimExpr offset = make_const(DataType::Int(32), 0)) const;
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
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const BufferNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = BufferNode;
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
  /*! \brief The shape of the buffer */
  Array<PrimExpr> shape;
  /*!
   * \brief The strides of each dimension
   *  This can be an empty array, indicating array is contiguous
   */
  Array<PrimExpr> strides;
  /*! \brief The offset in terms of number of dtype elements (including lanes) */
  PrimExpr elem_offset;
  // Meta data
  /*! \brief optional name of the buffer */
  std::string name;
  /*! \brief storage scope of the buffer, if other than global */
  std::string scope;
  /*! \brief Alignment requirement of data pointer in bytes. */
  int data_alignment;
  /*!
   * \brief Factor of elem_offset field,
   *  elem_offset is guaranteed to be multiple of offset_factor.
   */
  int offset_factor;
  /*! \brief buffer type */
  BufferType buffer_type;
  /*! \brief constructor */
  BufferNode() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
    v->Visit("strides", &strides);
    v->Visit("elem_offset", &elem_offset);
    v->Visit("name", &name);
    v->Visit("scope", &scope);
    v->Visit("data_alignment", &data_alignment);
    v->Visit("offset_factor", &offset_factor);
    v->Visit("buffer_type", &buffer_type);
  }

  /*! \return preferred index type for this buffer node */
  DataType DefaultIndexType() const {
    return shape.size() != 0 ? shape[0].dtype() : DataType::Int(32);
  }

  // User can specify data_alignment and offset_factor to be 0
  // A default value will be picked.
  TVM_DLL static Buffer make(Var ptr,
                             DataType dtype,
                             Array<PrimExpr> shape,
                             Array<PrimExpr> strides,
                             PrimExpr elem_offset,
                             std::string name,
                             std::string scope,
                             int data_alignment,
                             int offset_factor,
                             BufferType buffer_type);

  static constexpr const char* _type_key = "Buffer";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferNode, Object);
};

inline const BufferNode* Buffer::operator->() const {
  return static_cast<const BufferNode*>(get());
}

/*!
 * \brief Construct a new buffer given shape, and dtype.
 * \param shape The shape of the buffer,
 * \param dtype The content data type.
 * \param name The name of the buffer
 * \return The created buffer.
 * \sa BufferNode::make for complete constructor.
 */
TVM_DLL Buffer decl_buffer(Array<PrimExpr> shape,
                           DataType dtype = DataType::Float(32),
                           std::string name = "buffer");
}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_BUFFER_H_
