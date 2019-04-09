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
 * \file tvm/channel.h
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
  explicit Channel(NodePtr<Node> n) : NodeRef(n) {}
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
