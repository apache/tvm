/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Defines the base data structure
 */
#ifndef TVM_BASE_H_
#define TVM_BASE_H_

#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <string>
#include <memory>
#include <functional>
#include <typeinfo>
#include <type_traits>
#include <tvm/node.h>

namespace tvm {

using ::tvm::Node;
using ::tvm::NodeRef;
using ::tvm::AttrVisitor;

/*! \brief typedef the factory function of data iterator */
using NodeFactory = std::function<std::shared_ptr<Node> ()>;
/*!
 * \brief Registry entry for NodeFactory
 */
struct NodeFactoryReg
    : public dmlc::FunctionRegEntryBase<NodeFactoryReg,
                                        NodeFactory> {
};

#define TVM_REGISTER_NODE_TYPE(TypeName)                                \
  DMLC_REGISTRY_REGISTER(::tvm::NodeFactoryReg, NodeFactoryReg, TypeName) \
  .set_body([]() { return std::make_shared<TypeName>(); })

}  // namespace tvm
#endif  // TVM_BASE_H_
