
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
   * \brief construct a new buffer based on shape and strides.
   */
  explicit Buffer(Array<Expr> shape,
                  Type dtype = Float(32),
                  std::string name = "buffer");
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
  /*! \brief optional name of the buffer */
  std::string name;
  /*! \brief The pointer to the head of the data */
  Var data;
  /*! \brief The shape of the buffer */
  Array<Expr> shape;
  /*!
   * \brief The strides of each dimension
   *  This can be an empty array, indicating array is contiguous
   */
  Array<Expr> strides;
  /*! \brief data type in the content of the tensor */
  Type dtype;
  // Maybe need more information(alignment) later
  /*! \brief constructor */
  BufferNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("data", &data);
    v->Visit("shape", &shape);
    v->Visit("strides", &strides);
    v->Visit("dtype", &dtype);
  }

  static Buffer make(std::string name,
                     Var ptr,
                     Array<Expr> shape,
                     Array<Expr> strides,
                     Type dtype);

  static constexpr const char* _type_key = "Buffer";
  TVM_DECLARE_NODE_TYPE_INFO(BufferNode, Node);
};

inline const BufferNode* Buffer::operator->() const {
  return static_cast<const BufferNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_BUFFER_H_
