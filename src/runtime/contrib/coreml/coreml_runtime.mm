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
 * \file coreml_runtime.cc
 */
#include <tvm/runtime/registry.h>

#include "coreml_runtime.h"

namespace tvm {
namespace runtime {

MLModel *load_coreml_model(const std::string& model_path) {
  NSBundle* bundle = [NSBundle mainBundle];
  NSString* base = [bundle privateFrameworksPath];
  NSString* fname = [NSString stringWithUTF8String:("tvm/" + model_path).c_str()];
  NSString* assetPath = [base stringByAppendingPathComponent: fname];

  if (![[NSFileManager defaultManager] fileExistsAtPath:assetPath]) {
    assetPath = [NSString stringWithCString: model_path.c_str() encoding:NSUTF8StringEncoding];
  }

  NSURL *url = [NSURL fileURLWithPath:assetPath];

  MLModel *model = [MLModel modelWithContentsOfURL:url error:nil];
  if (model == nil) {
    NSLog(@"modelc %@ not found", url);
  }
  return model;
}

void CoreMLRuntime::Init(const std::string& model_path,
                         TVMContext ctx,
                         const std::vector<NSString *>& output_names) {
  model_ = load_coreml_model(model_path);
  ctx_ = ctx;
  input_dict_ = [NSMutableDictionary dictionary];
  output_names_ = output_names;
}

void CoreMLRuntime::Invoke() {
  id<MLFeatureProvider> input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict_ error:nil];
  output_ = [model_ predictionFromFeatures:input error:nil];
}

void CoreMLRuntime::SetInput(const std::string& key, DLTensor* data_in) {
  int64_t size = 1;
  NSMutableArray *shape = [[NSMutableArray alloc] init];
  for (int64_t i = 0; i < data_in->ndim; ++i) {
    size *= data_in->shape[i];
    [shape addObject:[NSNumber numberWithInteger:data_in->shape[i]]];
  }

  DataType dtype(data_in->dtype);
  MLMultiArrayDataType dataType;
  if (dtype == DataType::Float(64)) {
    dataType = MLMultiArrayDataTypeDouble;
    size *= sizeof(double);
  } else if (dtype == DataType::Float(32)) {
    dataType = MLMultiArrayDataTypeFloat32;
    size *= sizeof(float);
  } else {
    LOG(FATAL) << "unsupported data type " << dtype;
    return;
  }

  MLMultiArray *dest = [[MLMultiArray alloc] initWithShape:shape
                        dataType:dataType error:nil];

  CHECK(data_in->strides == NULL);
  memcpy(dest.dataPointer, data_in->data, size);

  NSString *nsKey = [NSString stringWithUTF8String:key.c_str()];
  [input_dict_ setObject:dest forKey:nsKey];
}

NDArray CoreMLRuntime::GetOutput(int index) const {
  NSString *name = output_names_[index];
  MLModelDescription *model_desc = model_.modelDescription;
  MLFeatureDescription *output_desc = model_desc.outputDescriptionsByName[name];
  MLMultiArrayConstraint *data_desc = output_desc.multiArrayConstraint;
  std::vector<int64_t> shape;
  int64_t size = 1;
  for (int64_t i = 0; i < data_desc.shape.count; ++i) {
    int n = data_desc.shape[i].intValue;
    size *= n;
    shape.push_back(n);
  }

  DataType dtype;
  if (data_desc.dataType == MLMultiArrayDataTypeDouble) {
    dtype = DataType::Float(64);
    size *= sizeof(double);
  } else if (data_desc.dataType == MLMultiArrayDataTypeFloat32) {
    dtype = DataType::Float(32);
    size *= sizeof(float);
  } else {
    LOG(FATAL) << "unexpected data type " << data_desc.dataType;
  }
  MLMultiArray *src = [output_ featureValueForName:name].multiArrayValue;
  NDArray ret = NDArray::Empty(shape, dtype, ctx_);
  ret.CopyFromBytes(src.dataPointer, size);

  return ret;
}

int CoreMLRuntime::GetNumOutputs() const {
  return output_names_.size();
}

PackedFunc CoreMLRuntime::GetFunction(
    const std::string& name,
    const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->Invoke();
      });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        const auto& input_name = args[0].operator std::string();
        this->SetInput(input_name, args[1]);
      });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetOutput(args[0]);
      });
  } else if (name == "get_num_outputs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetNumOutputs();
      });
  } else {
    return PackedFunc();
  }
}

Module CoreMLRuntimeCreate(const std::string& model_path,
                           TVMContext ctx,
                           const std::vector<NSString *>& output_names) {
  auto exec = make_object<CoreMLRuntime>();
  exec->Init(model_path, ctx, output_names);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.coreml_runtime.create")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
      std::vector<NSString *> output_names;
      for (size_t i = 2; i < args.size(); i++) {
        const std::string& name = args[i];
        output_names.push_back([NSString stringWithUTF8String:name.c_str()]);
      }
      *rv = CoreMLRuntimeCreate(args[0], args[1], output_names);
  });
}  // namespace runtime
}  // namespace tvm
