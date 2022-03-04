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
 * \file tvm/target/metadata_utils.cc
 * \brief Defines utility functions and classes for emitting metadata.
 */
#include "metadata_utils.h"

namespace tvm {
namespace codegen {

MetadataQueuer::MetadataQueuer(std::vector<QueueItem>* queue) : queue_{queue} {}

std::string address_from_parts(const std::vector<std::string>& parts) {
  std::stringstream ss;
  for (unsigned int i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << "_";
    }
    ss << parts[i];
  }
  return ss.str();
}

void MetadataQueuer::Visit(const char* key, double* value) {}
void MetadataQueuer::Visit(const char* key, int64_t* value) {}
void MetadataQueuer::Visit(const char* key, uint64_t* value) {}
void MetadataQueuer::Visit(const char* key, int* value) {}
void MetadataQueuer::Visit(const char* key, bool* value) {}
void MetadataQueuer::Visit(const char* key, std::string* value) {}
void MetadataQueuer::Visit(const char* key, DataType* value) {}
void MetadataQueuer::Visit(const char* key, runtime::NDArray* value) {}
void MetadataQueuer::Visit(const char* key, void** value) {}

void MetadataQueuer::Visit(const char* key, ObjectRef* value) {
  address_parts_.push_back(key);
  if (value->as<runtime::metadata::MetadataBaseNode>() != nullptr) {
    auto metadata = Downcast<runtime::metadata::MetadataBase>(*value);
    const runtime::metadata::MetadataArrayNode* arr =
        value->as<runtime::metadata::MetadataArrayNode>();
    std::cout << "Is array? " << arr << std::endl;
    if (arr != nullptr) {
      for (unsigned int i = 0; i < arr->array.size(); i++) {
        ObjectRef o = arr->array[i];
        std::cout << "queue-visiting array element " << i << ": " << o->type_index() << " ("
                  << o.operator->() << ")" << std::endl;
        if (o.as<runtime::metadata::MetadataBaseNode>() != nullptr) {
          std::stringstream ss;
          ss << i;
          address_parts_.push_back(ss.str());
          runtime::metadata::MetadataBase metadata = Downcast<runtime::metadata::MetadataBase>(o);
          ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
          address_parts_.pop_back();
        }
      }
    } else {
      ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
    }

    queue_->push_back(std::make_tuple(address_from_parts(address_parts_),
                                      Downcast<runtime::metadata::MetadataBase>(*value)));
  }
  address_parts_.pop_back();
}

}  // namespace codegen
}  // namespace tvm
