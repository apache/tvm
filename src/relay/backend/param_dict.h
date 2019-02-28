/*!
 *  Copyright (c) 2019 by Contributors
 * \file param_dict.h
 * \brief Definitions for serializing and deserializing parameter dictionaries.
 */
#ifndef TVM_RELAY_BACKEND_PARAM_DICT_H_
#define TVM_RELAY_BACKEND_PARAM_DICT_H_

#include <tvm/node/node.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <string>

namespace tvm {
namespace relay {

/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*!
 * \brief Wrapper node for naming `NDArray`s.
 */
struct NamedNDArrayNode : public ::tvm::Node {
  std::string name;
  tvm::runtime::NDArray array;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("array", &array);
  }

  static constexpr const char* _type_key = "NamedNDArray";
  TVM_DECLARE_NODE_TYPE_INFO(NamedNDArrayNode, Node);
};

TVM_DEFINE_NODE_REF(NamedNDArray, NamedNDArrayNode);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_PARAM_DICT_H_
