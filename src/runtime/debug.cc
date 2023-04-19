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
 * \file src/runtime/debug.cc
 * \brief Helpers for debugging at runtime.
 */

#include <tvm/runtime/debug.h>

namespace tvm {
namespace runtime {

template <typename T>
void AppendMembers(std::ostream& os, const NDArray& nd_array, int64_t dim0) {
  os << "=[";
  for (int64_t i = 0; i < dim0; ++i) {
    if (i > 0) {
      os << ",";
    }
    os << reinterpret_cast<T*>(nd_array->data)[i];
  }
  os << "]";
}

void AppendNDArray(std::ostream& os, const NDArray& nd_array, const DLDevice& host_device,
                   bool show_contents) {
  os << "NDArray[";
  os << "(";
  for (int dim = 0; dim < nd_array->ndim; ++dim) {
    if (dim > 0) {
      os << ",";
    }
    os << nd_array->shape[dim];
  }
  std::string basic_type = DLDataType2String(nd_array->dtype);
  os << ")," << basic_type;
  os << ",(" << nd_array->device.device_type;
  os << "," << nd_array->device.device_id;
  os << ")]";
  if (show_contents && nd_array->device.device_type == host_device.device_type &&
      nd_array->device.device_id == host_device.device_id) {
    int64_t dim0;
    if (nd_array->ndim == 0) {
      dim0 = 1;
    } else if (nd_array->ndim == 1) {
      dim0 = nd_array->shape[0];
      if (dim0 > 10) {
        // Too large.
        dim0 = 0;
      }
    } else {
      // Not rank-1.
      dim0 = 0;
    }
    if (dim0 > 0) {
      if (basic_type == "bool") {
        AppendMembers<bool>(os, nd_array, dim0);
      } else if (basic_type == "int8") {
        AppendMembers<int8_t>(os, nd_array, dim0);
      } else if (basic_type == "int16") {
        AppendMembers<int16_t>(os, nd_array, dim0);
      } else if (basic_type == "int32") {
        AppendMembers<int32_t>(os, nd_array, dim0);
      } else if (basic_type == "int64") {
        AppendMembers<int64_t>(os, nd_array, dim0);
      } else if (basic_type == "uint8") {
        AppendMembers<uint8_t>(os, nd_array, dim0);
      } else if (basic_type == "uint16") {
        AppendMembers<uint16_t>(os, nd_array, dim0);
      } else if (basic_type == "uint32") {
        AppendMembers<uint32_t>(os, nd_array, dim0);
      } else if (basic_type == "uint64") {
        AppendMembers<uint64_t>(os, nd_array, dim0);
      } else if (basic_type == "float32") {
        AppendMembers<float>(os, nd_array, dim0);
      } else if (basic_type == "float64") {
        AppendMembers<double>(os, nd_array, dim0);
      }
    }
  }
}

void AppendADT(std::ostream& os, const ADT& adt, const DLDevice& host_device, bool show_contents) {
  os << "ADT(" << adt->tag;
  for (size_t i = 0; i < adt->size; ++i) {
    os << ",";
    AppendRuntimeObject(os, adt[i], host_device, show_contents);
  }
  os << ")";
}

void AppendRuntimeObject(std::ostream& os, const ObjectRef& object, const DLDevice& host_device,
                         bool show_contents) {
  if (auto adt = object.as<ADT>()) {
    AppendADT(os, adt.value(), host_device, show_contents);
  } else if (auto nd_array_cont = object.as<NDArray>()) {
    AppendNDArray(os, nd_array_cont.value(), host_device, show_contents);
  } else {
    os << "?";
  }
}

std::string RuntimeObject2String(const ObjectRef& object, const DLDevice& host_device,
                                 bool show_contents) {
  std::ostringstream os;
  AppendRuntimeObject(os, object, host_device, show_contents);
  return os.str();
}

}  // namespace runtime
}  // namespace tvm
