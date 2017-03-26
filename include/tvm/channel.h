/*!
 *  Copyright (c) 2017 by Contributors
 * \file channel.h
 * \brief Channel object for pipeline.
 */
#ifndef TVM_CHANNEL_H_
#define TVM_CHANNEL_H_

#include <tvm/expr.h>

namespace tvm {
// Node container of channel
struct ChannelNode;

/*! \brief The data channel. */
class Channel : public NodeRef {
 public:
  /*! \brief default constructor  */
  Channel() {}
  explicit Channel(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const ChannelNode* operator->() const;
  // The container type
  using ContainerType = ChannelNode;
};

/*!
 * \brief Generalized FIFO channel.
 */
struct ChannelNode : public Node {
  /*! \brief Variable to channel handle */
  Var handle_var;
  /*! \brief default data type in read/write */
  Type dtype;
  // visit all attributes
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("handle_var", &handle_var);
    v->Visit("dtype", &dtype);
  }

  static Channel make(Var handle_var, Type dtype);
  static constexpr const char* _type_key = "Channel";

  TVM_DECLARE_NODE_TYPE_INFO(ChannelNode, Node);
};

// Inline implementations
inline const ChannelNode* Channel::operator->() const {
  return static_cast<const ChannelNode*>(node_.get());
}
}  // namespace tvm
#endif  // TVM_CHANNEL_H_
