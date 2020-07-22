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

void CoreMLModel::Invoke() {
  id<MLFeatureProvider> input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict_
                                                                                  error:nil];
  output_ = [model_ predictionFromFeatures:input error:nil];
}

void CoreMLModel::SetInput(const std::string& key, DLTensor* data_in) {
  int64_t size = 1;
  NSMutableArray* shape = [[NSMutableArray alloc] init];
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

  MLMultiArray* dest = [[MLMultiArray alloc] initWithShape:shape dataType:dataType error:nil];

  CHECK(data_in->strides == NULL);
  memcpy(dest.dataPointer, data_in->data, size);

  NSString* nsKey = [NSString stringWithUTF8String:key.c_str()];
  [input_dict_ setObject:dest forKey:nsKey];
}

NDArray CoreMLModel::GetOutput(int index) const {
  MLModelDescription* model_desc = model_.modelDescription;
  NSString* metadata = [model_desc metadata][MLModelDescriptionKey];
  NSData* data = [metadata dataUsingEncoding:NSUTF8StringEncoding];
  NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data
                                                       options:NSJSONReadingAllowFragments
                                                         error:nil];
  NSString* name = json[@"outputs"][index];
  MLFeatureDescription* output_desc = model_desc.outputDescriptionsByName[name];
  MLMultiArrayConstraint* data_desc = output_desc.multiArrayConstraint;
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
  MLMultiArray* src = [output_ featureValueForName:name].multiArrayValue;
  TVMContext cpu_ctx = {
      .device_type = kDLCPU,
      .device_id = 0,
  };
  NDArray ret = NDArray::Empty(shape, dtype, cpu_ctx);
  ret.CopyFromBytes(src.dataPointer, size);

  return ret;
}

int CoreMLModel::GetNumOutputs() const {
  MLModelDescription* model_desc = model_.modelDescription;
  return [[model_desc outputDescriptionsByName] count];
}

void CoreMLRuntime::Init(const std::string& _model_dir) {
  NSString* model_dir = [NSString stringWithUTF8String:(_model_dir).c_str()];
  if (![model_dir hasPrefix:@"/"]) {
    // find models in the bundle's framework
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* base = [bundle privateFrameworksPath];
    model_dir = [base stringByAppendingPathComponent:model_dir];
  }
  NSFileManager* fileMamager = [NSFileManager defaultManager];
  NSArray<NSString*>* files = [fileMamager contentsOfDirectoryAtPath:model_dir error:nil];
  for (NSString* file in files) {
    if ([[file pathExtension] isEqualToString:@"mlmodelc"]) {
      NSString* model_path = [model_dir stringByAppendingPathComponent:file];
      NSURL* url = [NSURL fileURLWithPath:model_path];
      const std::string& model_name = [[file stringByDeletingPathExtension] UTF8String];
      model_map_[model_name] = std::unique_ptr<CoreMLModel>(new CoreMLModel(url));
    }
  }
}

CoreMLModel& CoreMLRuntime::GetModel(const std::string& model_name) {
  CHECK(model_map_.count(model_name) > 0) << "No such model in this module: " << model_name;
  return *model_map_[model_name];
}

PackedFunc CoreMLRuntime::GetFunction(const std::string& name,
                                      const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "invoke") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { GetModel("main").Invoke(); });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      const auto& input_name = args[0].operator std::string();
      GetModel("main").SetInput(input_name, args[1]);
    });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = GetModel("main").GetOutput(args[0]);
    });
  } else if (name == "get_num_outputs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = GetModel("main").GetNumOutputs();
    });
  } else {
    // Return the packedfunc which executes the subgraph.
    return PackedFunc([sptr_to_self, name, this](TVMArgs args, TVMRetValue* rv) {
      CoreMLModel& model = GetModel(name);
      MLModelDescription* model_desc = [model.model_ modelDescription];
      NSString* metadata = [model_desc metadata][MLModelDescriptionKey];
      NSData* data = [metadata dataUsingEncoding:NSUTF8StringEncoding];
      NSDictionary* json = [NSJSONSerialization JSONObjectWithData:data
                                                           options:NSJSONReadingAllowFragments
                                                             error:nil];
      NSArray<NSString*>* input_names = json[@"inputs"];

      // Copy input tensors to corresponding data entries.
      for (auto i = 0; i < args.size() - 1; ++i) {
        CHECK(args[i].type_code() == kTVMDLTensorHandle || args[i].type_code() == kTVMNDArrayHandle)
            << "Expect NDArray or DLTensor as inputs\n";
        if (args[i].type_code() == kTVMDLTensorHandle) {
          model.SetInput([input_names[i] UTF8String], args[i]);
        } else {
          LOG(FATAL) << "Not implemented";
        }
      }

      // Execute the subgraph.
      model.Invoke();

      // TODO: Support multiple outputs.
      NDArray out = model.GetOutput(0);
      if (args[args.size() - 1].type_code() == kTVMDLTensorHandle) {
        DLTensor* arg = args[args.size() - 1];
        out.CopyTo(arg);
      } else {
        NDArray arg = args[args.size() - 1];
        out.CopyTo(arg);
      }
      *rv = out;
    });
  }
}

Module CoreMLRuntimeCreate(const std::string& model_dir) {
  auto exec = make_object<CoreMLRuntime>();
  exec->Init(model_dir);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.coreml_runtime.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CoreMLRuntimeCreate(args[0]);
});

void CoreMLRuntime::SaveToBinary(dmlc::Stream* stream) {
  stream->Write((uint32_t)model_map_.size());
  for (const auto& kv : model_map_) {
    const std::string& model_name = kv.first;
    NSURL* url = kv.second->url_;
    NSFileWrapper* dirWrapper = [[[NSFileWrapper alloc] initWithURL:url options:0
                                                              error:nil] autorelease];
    NSData* dirData = [dirWrapper serializedRepresentation];
    stream->Write(model_name);
    stream->Write((uint64_t)[dirData length]);
    stream->Write([dirData bytes], [dirData length]);
    LOG(INFO) << "Save " << model_name << " (" << [dirData length] << " bytes)";
  }
}

/*!
 * \brief Load a CoreML module from stream.
 *
 * \param strm The binary stream to load json.
 *
 * \return The created CoreML module.
 */
Module CoreMLRuntimeLoadFromBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);

  uint32_t nr_models;
  stream->Read(&nr_models);

  NSString* tempBaseDir = NSTemporaryDirectory();
  if (tempBaseDir == nil) tempBaseDir = @"/tmp";

  NSString* templateStr = [tempBaseDir stringByAppendingPathComponent:@"tvm.XXXXXX"];
  const char* fsTemplate = [templateStr fileSystemRepresentation];
  NSMutableData* bufferData = [NSMutableData dataWithBytes:fsTemplate
                                                    length:strlen(fsTemplate) + 1];
  char* buffer = (char*)[bufferData mutableBytes];
  char* result = mkdtemp(buffer);
  NSString* tempDir = [NSString stringWithUTF8String:result];

  for (int i = 0; i < nr_models; i++) {
    std::string model_name;
    stream->Read(&model_name);
    uint64_t length;
    stream->Read(&length);
    void* ptr = new char[length];
    stream->Read(ptr, length);
    NSData* data = [[NSData alloc] initWithBytesNoCopy:ptr length:length];
    NSFileWrapper* dirWrapper =
        [[[NSFileWrapper alloc] initWithSerializedRepresentation:data] autorelease];
    NSString* model_dir = [tempDir
        stringByAppendingPathComponent:[NSString stringWithUTF8String:(model_name + ".mlmodelc")
                                                                          .c_str()]];
    NSURL* url = [NSURL fileURLWithPath:model_dir];
    BOOL res = [dirWrapper writeToURL:url options:0 originalContentsURL:nil error:nil];
    CHECK(res) << "Failed to create model directory " << [model_dir UTF8String];
  }

  auto exec = make_object<CoreMLRuntime>();
  exec->Init([tempDir UTF8String]);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_coreml").set_body_typed(CoreMLRuntimeLoadFromBinary);

}  // namespace runtime
}  // namespace tvm
