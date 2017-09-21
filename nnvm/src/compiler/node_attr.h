/*!
 * Copyright (c) 2017 by Contributors
 * \file node_attr.h
 * \brief utility to access node attributes
*/
#ifndef NNVM_COMPILER_NODE_ATTR_H_
#define NNVM_COMPILER_NODE_ATTR_H_

#include <nnvm/op.h>
#include <nnvm/compiler/op_attr_types.h>
#include <unordered_map>
#include <string>

namespace nnvm {
namespace compiler {

using AttrDict = std::unordered_map<std::string, std::string>;
/*!
 * \brief Get canonicalized attr dict from node
 * \param attrs The node attrs
 * \return The attribute dict
 */
inline AttrDict GetAttrDict(const NodeAttrs& attrs) {
  static auto& fgetdict = nnvm::Op::GetAttr<FGetAttrDict>("FGetAttrDict");
  if (fgetdict.count(attrs.op)) {
    return fgetdict[attrs.op](attrs);
  } else {
    return attrs.dict;
  }
}

}  // namespace compiler
}  // namespace nnvm
#endif  // NNVM_COMPILER_NODE_ATTR_H_
