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
#include <tvm/meta_schedule/utils.h>

namespace tvm {
namespace meta_schedule {

/******** ArgInfo ********/

ArgInfo ArgInfo::FromJSON(const ObjectRef& json_obj) {
  // The JSON object is always an array whose first element is a tag. For example:
  // `['TENSOR', 'float32', [1, 224, 224, 3]]
  // Step 1. Extract the tag
  String tag{runtime::ObjectPtr<runtime::StringObj>(nullptr)};
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() >= 1);
    tag = Downcast<String>(json_array->at(0));
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  // Step 2. Dispatch the tag to corresponding subclass of ArgInfo
  if (tag == "TENSOR") {
    return TensorInfo::FromJSON(json_obj);
  }
  LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj;
  throw;
}

Array<ArgInfo> ArgInfo::FromPrimFunc(const tir::PrimFunc& func) {
  using support::AsVector;
  Array<ArgInfo> result;
  result.reserve(func->params.size());
  for (const tir::Var& arg : func->params) {
    if (Optional<tir::Buffer> _buffer = func->buffer_map.Get(arg)) {
      tir::Buffer buffer = _buffer.value();
      result.push_back(TensorInfo(/*dtype=*/buffer->dtype,
                                  /*shape=*/AsVector<PrimExpr, int64_t>(buffer->shape)));
    } else {
      LOG(FATAL) << "ValueError: Unsupported argument type: " << arg;
    }
  }
  return result;
}

/******** TensorInfo ********/

TensorInfo::TensorInfo(runtime::DataType dtype, runtime::ShapeTuple shape) {
  ObjectPtr<TensorInfoNode> n = make_object<TensorInfoNode>();
  n->dtype = dtype;
  n->shape = shape;
  this->data_ = std::move(n);
}

ObjectRef TensorInfoNode::AsJSON() const {
  static String tag = "TENSOR";
  String dtype = DLDataType2String(this->dtype);
  Array<Integer> shape = support::AsArray(this->shape);
  return Array<ObjectRef>{tag, dtype, shape};
}

TensorInfo TensorInfo::FromJSON(const ObjectRef& json_obj) {
  DLDataType dtype;
  Array<Integer> shape;
  try {
    const ArrayNode* json_array = json_obj.as<ArrayNode>();
    CHECK(json_array && json_array->size() == 3);
    // Load json[1] => dtype
    {
      String dtype_str = Downcast<String>(json_array->at(1));
      dtype = runtime::String2DLDataType(dtype_str);
    }
    // Load json[2] => shape
    shape = Downcast<Array<Integer>>(json_array->at(2));
  } catch (const std::runtime_error& e) {  // includes tvm::Error and dmlc::Error
    LOG(FATAL) << "ValueError: Unable to parse the JSON object: " << json_obj
               << "\nThe error is: " << e.what();
  }
  return TensorInfo(DataType(dtype), ShapeTuple(shape.begin(), shape.end()));
}

/******** Repr ********/

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorInfoNode>([](const ObjectRef& n, ReprPrinter* p) {
      const auto* self = n.as<TensorInfoNode>();
      ICHECK(self);
      p->stream << "TensorInfo(\"" << self->dtype << "\", " << self->shape << ")";
    });

/******** FFI ********/

TVM_REGISTER_OBJECT_TYPE(ArgInfoNode);
TVM_REGISTER_NODE_TYPE(TensorInfoNode);

TVM_REGISTER_GLOBAL("meta_schedule.ArgInfoAsJSON").set_body_method<ArgInfo>(&ArgInfoNode::AsJSON);
TVM_REGISTER_GLOBAL("meta_schedule.ArgInfoFromPrimFunc").set_body_typed(ArgInfo::FromPrimFunc);
TVM_REGISTER_GLOBAL("meta_schedule.ArgInfoFromJSON").set_body_typed(ArgInfo::FromJSON);
TVM_REGISTER_GLOBAL("meta_schedule.TensorInfo")
    .set_body_typed([](runtime::DataType dtype, runtime::ShapeTuple shape) -> TensorInfo {
      return TensorInfo(dtype, shape);
    });

}  // namespace meta_schedule
}  // namespace tvm
