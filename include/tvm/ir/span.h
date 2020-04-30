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
 * \file tvm/ir/span.h
 * \brief Span information for debugging purposes.
 */
#ifndef TVM_IR_SPAN_H_
#define TVM_IR_SPAN_H_

#include <tvm/runtime/object.h>
#include <tvm/node/node.h>
#include <string>

namespace tvm {
/*!
 * \brief The source name in the Span
 * \sa SourceNameNode, Span
 */
class SourceName;
/*!
 * \brief The name of a source fragment.
 */
class SourceNameNode : public Object {
 public:
  /*! \brief The source name. */
  String name;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  bool SEqualReduce(const SourceNameNode* other, SEqualReducer equal) const {
    return equal(name, other->name);
  }

  static constexpr const char* _type_key = "SourceName";
  TVM_DECLARE_FINAL_OBJECT_INFO(SourceNameNode, Object);
};

/*!
 * \brief The source name of a file span.
 * \sa SourceNameNode, Span
 */
class SourceName : public ObjectRef {
 public:
  /*!
   * \brief Get an SourceName for a given operator name.
   *  Will raise an error if the source name has not been registered.
   * \param name Name of the operator.
   * \return SourceName valid throughout program lifetime.
   */
  TVM_DLL static SourceName Get(const String& name);

  TVM_DEFINE_OBJECT_REF_METHODS(SourceName, ObjectRef, SourceNameNode);
};

/*!
 * \brief Span information for debugging purposes
 */
class Span;
/*!
 * \brief Stores locations in frontend source that generated a node.
 */
class SpanNode : public Object {
 public:
  /*! \brief The source name */
  SourceName source;
  /*! \brief Line number */
  int lineno;
  /*! \brief column offset */
  int col_offset;
  // override attr visitor
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("source", &source);
    v->Visit("lineno", &lineno);
    v->Visit("col_offset", &col_offset);
  }

  bool SEqualReduce(const SpanNode* other, SEqualReducer equal) const {
    return
        equal(source, other->source) &&
        equal(lineno, other->lineno) &&
        equal(col_offset, other->col_offset);
  }

  TVM_DLL static Span make(SourceName source, int lineno, int col_offset);

  static constexpr const char* _type_key = "Span";
  TVM_DECLARE_FINAL_OBJECT_INFO(SpanNode, Object);
};


class Span : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Span, ObjectRef, SpanNode);
};

}  // namespace tvm
#endif  // TVM_IR_SPAN_H_
