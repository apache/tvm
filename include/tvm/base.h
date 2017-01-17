/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Defines the base data structure
 */
#ifndef TVM_BASE_H_
#define TVM_BASE_H_

#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <tvm/node.h>
#include <string>
#include <memory>
#include <functional>
#include <typeinfo>
#include <type_traits>

namespace tvm {

/*!
 *\brief whether to use CUDA runtime
 */
#ifndef TVM_CUDA_RUNTIME
#define TVM_CUDA_RUNTIME 1
#endif

/*!
 *\brief whether to use opencl runtime
 */
#ifndef TVM_OPENCL_RUNTIME
#define TVM_OPENCL_RUNTIME 0
#endif

using ::tvm::Node;
using ::tvm::NodeRef;
using ::tvm::AttrVisitor;

/*!
 * \brief save the node as well as all the node it depends on as json.
 *  This can be used to serialize any TVM object
 *
 * \return the string representation of the node.
 */
std::string SaveJSON(const NodeRef& node);

/*!
 * \brief Internal implementation of LoadJSON
 * Load tvm Node object from json and return a shared_ptr of Node.
 * \param json_str The json string to load from.
 *
 * \return The shared_ptr of the Node.
 */
std::shared_ptr<Node> LoadJSON_(std::string json_str);

/*!
 * \brief Load the node from json string.
 *  This can be used to deserialize any TVM object.
 *
 * \param json_str The json string to load from.
 *
 * \tparam NodeType the nodetype
 *
 * \code
 *  Expr e = LoadJSON<Expr>(json_str);
 * \endcode
 */
template<typename NodeType,
         typename = typename std::enable_if<std::is_base_of<NodeRef, NodeType>::value>::type >
inline NodeType LoadJSON(const std::string& json_str) {
  return NodeType(LoadJSON_(json_str));
}

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
  static DMLC_ATTRIBUTE_UNUSED ::tvm::NodeFactoryReg & __make_Node ## _ ## TypeName ## __ = \
      ::dmlc::Registry<::tvm::NodeFactoryReg>::Get()->__REGISTER__(TypeName::_type_key) \
      .set_body([]() { return std::make_shared<TypeName>(); })

}  // namespace tvm
#endif  // TVM_BASE_H_
