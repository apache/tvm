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
#ifndef TVM_RELAY_PRINTER_META_DATA_H_
#define TVM_RELAY_PRINTER_META_DATA_H_

#include <tvm/node/serialization.h>

#include <string>
#include <unordered_map>

#include "doc.h"

namespace tvm {
namespace relay {
/*!
 * \brief Meta data context for Printers
 *
 * This is an important part to enable bi-directional serializability.
 * We use tvm's Node system to build the current IR.
 * It can be hard to design a text format for all the possible nodes
 * as the set of nodes can grow when we do more extensions.
 *
 * Instead of trying to design readable text format for every node,
 * we support a meta data section in the text format.
 * We allow the text format to refer to a node in the meta data section.
 *
 * The meta data section is a json serialized string of an Map<string, Array<NodeRef>>.
 * Each element in the meta data section can be referenced by the text format.
 * Each meta data node is printed in the following format.
 *
 * meta[type-key-of-node>][<index-in-meta-section>]
 *
 * Specifically, consider the following IR(constructed by python).
 *
 * \code
 *
 * n = tvm.var("n")
 * x = tvm.relay.var("x", shape=(n, 1))
 * f = tvm.relay.Function([x], x)
 * print(f.astext())
 *
 * \endcode
 *
 * The corresponding text format is shown in the following code block.
 *
 * \code
 *
 * fn (%x: Tensor[(meta[Variable][0],), float32]) {
 *   %x
 * }
 * # Meta data section is a json-serialized string
 * # of the following array.
 * # [tvm.var("n")]
 *
 * \endcode
 *
 * Note that we store tvm.var("n") in the meta data section.
 * Since it is stored in the index-0 in the meta data section,
 * we print it as meta[Variable][0].
 *
 * The text parser can recover this object by loading from the corresponding
 * location in the meta data section.
 *
 * This is a design trade-off.
 * It allows us to embedded any meta data in the text format,
 * while still being able to tweak the text part of the printed IR easily.
 */
class TextMetaDataContext {
 public:
  /*!
   * \brief Get text representation of meta node.
   * \param node The node to be converted to meta node.
   * \return A string representation of the meta node.
   */
  Doc GetMetaNode(const ObjectRef& node) {
    auto it = meta_repr_.find(node);
    if (it != meta_repr_.end()) {
      return it->second;
    }
    std::string type_key = node->GetTypeKey();
    ICHECK(!type_key.empty());
    Array<ObjectRef>& mvector = meta_data_[type_key];
    int64_t index = static_cast<int64_t>(mvector.size());
    mvector.push_back(node);
    Doc doc;
    doc << "meta[" << type_key << "][" << index << "]";
    meta_repr_[node] = doc;
    return meta_repr_[node];
  }

  /*!
   * \brief Test whether a node has been put in meta
   * \param node The query node
   * \return whether the node has been put in meta
   */
  bool InMeta(const ObjectRef& node) { return meta_repr_.find(node) != meta_repr_.end(); }

  /*!
   * \brief Print a key value pair
   */
  Doc PrintKeyValue(const std::string& str, const Doc& v) const {
    return Doc() << "\"" << str << "\": " << v;
  }

  /*!
   * \brief Get the metadata section in json format.
   * \return the meta data string.
   */
  Doc GetMetaSection() const {
    if (meta_data_.size() == 0) return Doc();
    return Doc::RawText(SaveJSON(Map<String, ObjectRef>(meta_data_.begin(), meta_data_.end())));
  }

  /*! \return whether the meta data context is empty. */
  bool empty() const { return meta_data_.empty(); }

 private:
  /*! \brief additional metadata stored in TVM json format */
  std::unordered_map<String, Array<ObjectRef>> meta_data_;
  /*! \brief map from meta data into its string representation */
  std::unordered_map<ObjectRef, Doc, ObjectPtrHash, ObjectPtrEqual> meta_repr_;
};
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PRINTER_META_DATA_H_
