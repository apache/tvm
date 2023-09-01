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
 * Printer implementation for containers
 * \file node/container_printint.cc
 */
#include <tvm/node/functor.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/registry.h>

namespace tvm {

// Container printer
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ArrayNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ArrayNode*>(node.get());
      p->stream << '[';
      for (size_t i = 0; i < op->size(); ++i) {
        if (i != 0) {
          p->stream << ", ";
        }
        p->Print(op->at(i));
      }
      p->stream << ']';
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MapNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const MapNode*>(node.get());
      p->stream << '{';
      for (auto it = op->begin(); it != op->end(); ++it) {
        if (it != op->begin()) {
          p->stream << ", ";
        }
        if (it->first->IsInstance<StringObj>()) {
          p->stream << '\"' << Downcast<String>(it->first) << "\": ";
        } else {
          p->Print(it->first);
          p->stream << ": ";
        }
        p->Print(it->second);
      }
      p->stream << '}';
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShapeTupleObj>([](const ObjectRef& node, ReprPrinter* p) {
      p->stream << Downcast<ShapeTuple>(node);
    });
}  // namespace tvm
