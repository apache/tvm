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
namespace metadata {

std::string AddressFromParts(const std::vector<std::string>& parts) {
  std::stringstream ss;
  for (unsigned int i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      ss << "_";
    }
    ss << parts[i];
  }
  return ss.str();
}

DiscoverArraysVisitor::DiscoverArraysVisitor(std::vector<DiscoveredArray>* queue) : queue_{queue} {}

void DiscoverArraysVisitor::Visit(const char* key, double* value) {}
void DiscoverArraysVisitor::Visit(const char* key, int64_t* value) {}
void DiscoverArraysVisitor::Visit(const char* key, uint64_t* value) {}
void DiscoverArraysVisitor::Visit(const char* key, int* value) {}
void DiscoverArraysVisitor::Visit(const char* key, bool* value) {}
void DiscoverArraysVisitor::Visit(const char* key, std::string* value) {}
void DiscoverArraysVisitor::Visit(const char* key, DataType* value) {}
void DiscoverArraysVisitor::Visit(const char* key, runtime::NDArray* value) {}
void DiscoverArraysVisitor::Visit(const char* key, void** value) {}

void DiscoverArraysVisitor::Visit(const char* key, ObjectRef* value) {
  address_parts_.push_back(key);
  if (value->as<runtime::metadata::MetadataBaseNode>() != nullptr) {
    auto metadata = Downcast<runtime::metadata::MetadataBase>(*value);
    const runtime::metadata::MetadataArrayNode* arr =
        value->as<runtime::metadata::MetadataArrayNode>();
    if (arr != nullptr) {
      for (unsigned int i = 0; i < arr->array.size(); i++) {
        ObjectRef o = arr->array[i];
        if (o.as<runtime::metadata::MetadataBaseNode>() != nullptr) {
          std::stringstream ss;
          ss << i;
          address_parts_.push_back(ss.str());
          runtime::metadata::MetadataBase metadata = Downcast<runtime::metadata::MetadataBase>(o);
          ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
          address_parts_.pop_back();
        }
      }

      queue_->push_back(std::make_tuple(AddressFromParts(address_parts_),
                                        Downcast<runtime::metadata::MetadataArray>(metadata)));
    } else {
      ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
    }
  }
  address_parts_.pop_back();
}

void DiscoverComplexTypesVisitor::Visit(const char* key, double* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, int64_t* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, uint64_t* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, int* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, bool* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, std::string* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, DataType* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, runtime::NDArray* value) {}
void DiscoverComplexTypesVisitor::Visit(const char* key, void** value) {}

bool DiscoverComplexTypesVisitor::DiscoverType(std::string type_key) {
  VLOG(2) << "DiscoverType " << type_key;
  auto position_it = type_key_to_position_.find(type_key);
  if (position_it != type_key_to_position_.end()) {
    return false;
  }

  queue_->emplace_back(tvm::runtime::metadata::MetadataBase());
  type_key_to_position_[type_key] = queue_->size() - 1;
  return true;
}

void DiscoverComplexTypesVisitor::DiscoverInstance(runtime::metadata::MetadataBase md) {
  auto position_it = type_key_to_position_.find(md->GetTypeKey());
  ICHECK(position_it != type_key_to_position_.end())
      << "DiscoverInstance requires that DiscoverType has already been called: type_key="
      << md->GetTypeKey();

  int queue_position = (*position_it).second;
  if (!(*queue_)[queue_position].defined() && md.defined()) {
    VLOG(2) << "DiscoverInstance  " << md->GetTypeKey() << ":" << md;
    (*queue_)[queue_position] = md;
  }
}

void DiscoverComplexTypesVisitor::Visit(const char* key, ObjectRef* value) {
  ICHECK_NOTNULL(value->as<runtime::metadata::MetadataBaseNode>());

  auto metadata = Downcast<runtime::metadata::MetadataBase>(*value);
  const runtime::metadata::MetadataArrayNode* arr =
      value->as<runtime::metadata::MetadataArrayNode>();

  if (arr == nullptr) {
    VLOG(2) << "No array, object-traversing " << metadata->GetTypeKey();
    ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
    DiscoverType(metadata->GetTypeKey());
    DiscoverInstance(metadata);
    return;
  }

  if (arr->kind != tvm::runtime::metadata::MetadataKind::kMetadata) {
    return;
  }

  bool needs_instance = DiscoverType(arr->type_key);
  for (unsigned int i = 0; i < arr->array.size(); i++) {
    tvm::runtime::metadata::MetadataBase o =
        Downcast<tvm::runtime::metadata::MetadataBase>(arr->array[i]);
    if (needs_instance) {
      DiscoverInstance(o);
      needs_instance = false;
    }
    ReflectionVTable::Global()->VisitAttrs(o.operator->(), this);
  }
}

void DiscoverComplexTypesVisitor::Discover(runtime::metadata::MetadataBase metadata) {
  ReflectionVTable::Global()->VisitAttrs(metadata.operator->(), this);
  DiscoverType(metadata->GetTypeKey());
  DiscoverInstance(metadata);
}

}  // namespace metadata
}  // namespace codegen
}  // namespace tvm
