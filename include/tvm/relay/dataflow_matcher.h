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
 * \file tvm/relay/dataflow_matcher.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAY_DATAFLOW_MATCHER_H_
#define TVM_RELAY_DATAFLOW_MATCHER_H_

#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/dataflow_pattern_functor.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace relay {

class DFPatternCallback;
/*!
 * \brief Base type of all dataflow pattern callbacks.
 * \sa DFPatternCallback
 */
class DFPatternCallbackNode : public Object {
 public:
  /*! \brief Pattern this callback matches */
  DFPattern pattern_;
  /*! \brief Function to call when finding a matched expression */
  PackedFunc function_;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "DFPatternCallbackNode";
  TVM_DECLARE_BASE_OBJECT_INFO(DFPatternCallbackNode, Object);
};

/*!
 * \brief Managed reference to dataflow pattern callbacks.
 * \sa DFPatternCallbackNode
 */
class DFPatternCallback : public ObjectRef {
 public:
  TVM_DLL DFPatternCallback(DFPattern pattern, PackedFunc callback);
  TVM_DEFINE_OBJECT_REF_METHODS(DFPatternCallback, ObjectRef, DFPatternCallbackNode);
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_DATAFLOW_MATCHER_H_
