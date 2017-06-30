/*!
 *  Copyright (c) 2016 by Contributors
 * \file buffer.h
 * \brief Symbolic n-dimensional array, to represent a memory buffer.
 */
#ifndef TVM_BUFFER_H_
#define TVM_BUFFER_H_

#include <tvm/container.h>
#include <string>

#include "./base.h"
#include "./expr.h"

namespace tvm {

// Internal node container Buffer
class BufferNode;
/*!
 * \brief Buffer is a symbolic n-darray structure.
 *  It is a composition of primitive symbolic types,
 *  used to specify input/output strcuture of the program.
 */
class Buffer : public NodeRef {
 public:
  Buffer() {}
  explicit Buffer(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief Generate a load expression loading the index location of buffer.
   * \param index The index to the buffer.
   * \return The load expression.
   */
  Expr MakeLoad(Array<Expr> index) const;
  /*!
   * \brief Generate a store statement.
   * \param index The index to the buffer.
   * \param value The value to be stored.
   * \return The load expression.
   */
  Stmt MakeStore(Array<Expr> index, Expr value) const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const BufferNode* operator->() const;

  /*! \brief specify container node */
  using ContainerType = BufferNode;
};

/*! \brief Node to represent a buffer */
class BufferNode : public Node {
 public:
  // Data fields.
  /*! \brief The pointer to the head of the data */
  Var data;
  /*! \brief data type in the content of the tensor */
  Type dtype;
  /*! \brief The shape of the buffer */
  Array<Expr> shape;
  /*!
   * \brief The strides of each dimension
   *  This can be an empty array, indicating array is contiguous
   */
  Array<Expr> strides;
  /*!
   * \brief The offset in bytes to the beginning pointer to data
   *  Can be undefined, indicating this must be zero.
   */
  Expr byte_offset;
  // Meta data
  /*! \brief optional name of the buffer */
  std::string name;
  /*! \brief storage scope of the buffer, if other than global */
  std::string scope;
  /*! \brief Alignment bytes size of byte_offset */
  int offset_alignment;
  /*! \brief constructor */
  BufferNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("data", &data);
    v->Visit("dtype", &dtype);
    v->Visit("shape", &shape);
    v->Visit("strides", &strides);
    v->Visit("byte_offset", &byte_offset);
    v->Visit("name", &name);
    v->Visit("scope", &scope);
    v->Visit("offset_alignment", &offset_alignment);
  }

  static Buffer make(Var ptr,
                     Type dtype,
                     Array<Expr> shape,
                     Array<Expr> strides,
                     Expr byte_offset,
                     std::string name,
                     std::string scope,
                     int offset_alignment);

  static constexpr const char* _type_key = "Buffer";
  TVM_DECLARE_NODE_TYPE_INFO(BufferNode, Node);
};

inline const BufferNode* Buffer::operator->() const {
  return static_cast<const BufferNode*>(node_.get());
}

/*!
 * \brief Construct a new buffer given shape, and dtype.
 * \param shape The shape of the buffer,
 * \param dtype The content data type.
 * \param name The name of the buffer
 * \return The created buffer.
 * \sa BufferNode::make for complete constructor.
 */
Buffer decl_buffer(Array<Expr> shape,
                   Type dtype = Float(32),
                   std::string name = "buffer");
}  // namespace tvm
#endif  // TVM_BUFFER_H_
