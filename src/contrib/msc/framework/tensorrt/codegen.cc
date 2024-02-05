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
 * \file src/contrib/msc/framework/tensorrt/codegen.cc
 * \brief Codegen related classes.
 */

#include "codegen.h"

#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>

#include <set>

#include "../../core/codegen/codegen_json.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::relax;

void TensorRTCodeGen::CodeGenClassDeclare() {
  stack_.line("#include \"NvInfer.h\"")
      .line("#include \"NvInferRuntimeCommon.h\"")
      .line("#include \"utils/base.h\"")
      .line("#include \"utils/trt_common.h\"");
  if (config()->precision == "int8") {
    stack_.line("#include \"utils/trt_quantize.h\"");
  }
  // plugin headers
  if (config()->use_plugin) {
    std::set<String> plugins;
    for (const auto& n : graph()->node_names) {
      const auto& node = graph()->FindNode(n);
      if (IsPlugin(node->optype) && !plugins.count(node->optype)) {
        stack_.line("#include \"plugin/" + node->optype + "_op.h\"");
        plugins.insert(node->optype);
      }
    }
  }
  stack_.line().line("using namespace nvinfer1;").line();
  StartNamespace();
  // start class declare
  stack_.class_def(graph()->name).class_start().scope_start("public:");
  // declare build method
  stack_.func_def("Build", "bool")
      .func_arg("builder", "TRTPtr<IBuilder>&")
      .func_arg("network", "TRTPtr<INetworkDefinition>&");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.func_arg("config", "TRTPtr<IBuilderConfig>&");
  }
  stack_.func_arg("logger", "TRTLogger&").func_start().func_end();
  // define cleanup method
  stack_.func_def("CleanUp", "bool")
      .func_start()
      .for_start("mem", "mWeights")
      .func_call("free")
      .call_arg("(void*) (mem.second.values)")
      .for_end()
      .func_end("true");
  // end public scope
  stack_.scope_end();
  // private scope
  stack_.scope_start("private:").declare("std::map<std::string, Weights>", "mWeights").scope_end();
  // end class declare
  stack_.class_end();
  // declare test function
  stack_.func_def("test_" + graph()->name, "bool")
      .func_arg("engine", "std::shared_ptr<ICudaEngine>&")
      .func_arg("reader", "DatasetReader&")
      .func_arg("logger", "TRTLogger&")
      .func_start()
      .func_end();
  EndNamespace();
}

void TensorRTCodeGen::CodeGenClassDefine() {
  auto malloc_buffer = [this](const MSCTensor& tensor) {
    const String& idx_var = "idx_" + IdxTensor(tensor);
    this->stack_
        .func_call("getBindingIndex", DocUtils::ToDeclare("int", idx_var),
                   DocUtils::ToPtr("engine"))
        .call_arg(DocUtils::ToStr(tensor->name))
        .func_call("CHECK")
        .func_call("cudaMalloc")
        .call_arg(DocUtils::ToIndex("&gpu_buffers", idx_var))
        .call_arg(GetTensorBytes(tensor))
        .pop_nest()
        .func_call("malloc", DocUtils::ToIndex("cpu_buffers", idx_var))
        .call_arg(GetTensorBytes(tensor));
  };
  stack_.line("#include \"" + graph()->name + ".h\"").line();
  StartNamespace();
  // start define build method
  stack_.func_def(graph()->name + "::Build", "bool")
      .func_arg("builder", "TRTPtr<IBuilder>&")
      .func_arg("network", "TRTPtr<INetworkDefinition>&");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.func_arg("config", "TRTPtr<IBuilderConfig>&");
  }
  stack_.func_arg("logger", "TRTLogger&").func_start();
  // save codegen before build
  if (config()->use_tools) {
    const auto* pf = runtime::Registry::Get("msc_tool.codegen_step");
    ICHECK(pf != nullptr) << "Cannot find msc_tool.codegen_step func.";
    before_build_codes_ = (*pf)(GetStepCtx(), "before_build", graph()->name, config()->tools_tag);
  }
  if (graph()->weight_holders.size() > 0) {
    stack_.func_call("TRTUtils::LoadWeights", "mWeights")
        .call_arg(DocUtils::ToStr(graph()->name + ".wts"));
  }
  // build layers
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    CodeGenNode(node, config()->use_tools);
  }
  // mark outputs
  stack_.comment("Mark outputs");
  for (const auto& o : graph()->GetOutputs()) {
    const auto& pair = graph()->FindProducerAndIdx(o);
    stack_.func_call("markOutput", NullOpt, DocUtils::ToPtr("network"))
        .call_arg("*" + IdxOutputBase(pair.first, pair.second));
  }
  // mark batch_size
  stack_.comment("Mark batch size");
  stack_.func_call("createOptimizationProfile", DocUtils::ToDeclare("auto", "profile"),
                   DocUtils::ToPtr("builder"));
  Array<String> batch_flags{"MIN", "MAX", "OPT"};
  for (const auto& i : graph()->GetInputs()) {
    for (const auto& f : batch_flags) {
      stack_.func_call("setDimensions", NullOpt, DocUtils::ToPtr("profile"))
          .call_arg(DocUtils::ToStr(i->name))
          .call_arg("OptProfileSelector::k" + f)
          .call_arg(ToDims(i->shape));
    }
  }
  // set max workspace
  stack_.comment("Set max worksapce");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.func_call("setMaxWorkspaceSize", NullOpt, DocUtils::ToPtr("config"))
        .call_arg(config()->max_workspace);
  } else {
    stack_.func_call("setMaxWorkspaceSize", NullOpt, DocUtils::ToPtr("builder"))
        .call_arg(config()->max_workspace);
  }
  // set data type
  if (config()->precision == "float16") {
    stack_.comment("Set network precision")
        .cond_if("!builder->platformHasFastFp16()")
        .func_call("log", "", "logger")
        .call_arg("ILogger::Severity::kINTERNAL_ERROR")
        .call_arg(DocUtils::ToStr("platform do not support float16, fallback to float32"))
        .cond_else()
        .func_call("setFlag", NullOpt, DocUtils::ToPtr("config"))
        .call_arg("BuilderFlag::kFP16");
    if (config()->precision_mode == "strict") {
      stack_.func_call("setFlag", NullOpt, DocUtils::ToPtr("config"))
          .call_arg("BuilderFlag::kSTRICT_TYPES");
    }
    stack_.func_call("log", "", "logger")
        .call_arg("ILogger::Severity::kINFO")
        .call_arg(DocUtils::ToStr("use float16 to build the engine"))
        .cond_end();
  } else if (config()->precision == "int8") {
    stack_.comment("Set network precision")
        .cond_if("!builder->platformHasFastInt8()")
        .func_call("log", "", "logger")
        .call_arg("ILogger::Severity::kINTERNAL_ERROR")
        .call_arg(DocUtils::ToStr("platform do not support int8, fallback to float32"))
        .cond_else()
        .func_call("setFlag", NullOpt, DocUtils::ToPtr("config"))
        .call_arg("BuilderFlag::kINT8");
    if (config()->precision_mode == "strict") {
      stack_.func_call("setFlag", NullOpt, DocUtils::ToPtr("config"))
          .call_arg("BuilderFlag::kSTRICT_TYPES");
    } else if (config()->precision_mode == "prefer") {
      stack_.func_call("setFlag", NullOpt, DocUtils::ToPtr("config"))
          .call_arg("BuilderFlag::kPREFER_PRECISION_CONSTRAINTS");
    } else if (config()->precision_mode == "obey") {
      stack_.func_call("setFlag", NullOpt, DocUtils::ToPtr("config"))
          .call_arg("BuilderFlag::kOBEY_PRECISION_CONSTRAINTS");
    }
    stack_.func_call("log", "", "logger")
        .call_arg("ILogger::Severity::kINFO")
        .call_arg(DocUtils::ToStr("use int8 to build the engine"))
        .cond_end();
  }
  // save codegen after build
  if (config()->use_tools) {
    const auto* pf = runtime::Registry::Get("msc_tool.codegen_step");
    ICHECK(pf != nullptr) << "Cannot find msc_tool.codegen_step func.";
    after_build_codes_ = (*pf)(GetStepCtx(), "after_build", graph()->name, config()->tools_tag);
  }
  // end define build method
  stack_.func_end("true");
  // start define test function
  stack_.func_def("test_" + graph()->name, "bool")
      .func_arg("engine", "std::shared_ptr<ICudaEngine>&")
      .func_arg("reader", "DatasetReader&")
      .func_arg("logger", "TRTLogger&")
      .func_start();
  stack_.comment("Create context")
      .func_call("TRTPtr<IExecutionContext>", DocUtils::ToDeclare("auto", "context"))
      .func_call("createExecutionContext", NullOpt, DocUtils::ToPtr("engine"))
      .pop_nest();
  ReturnOnFail("context", "Failed to create the context");
  // prepare variables
  stack_.declare("bool", "pass", 0, false)
      .declare_arg("true")
      .declare("cudaStream_t", "stream")
      .func_call("CHECK")
      .func_call("cudaStreamCreate")
      .call_arg("&stream")
      .pop_nest();
  // malloc buffers
  size_t binding_num = graph()->input_names.size() + graph()->output_names.size();
  stack_.comment("Malloc and copy the buffers")
      .declare("void*", "cpu_buffers", binding_num)
      .declare("void*", "gpu_buffers", binding_num);
  for (const auto& i : graph()->GetInputs()) {
    malloc_buffer(i);
  }
  for (const auto& o : graph()->GetOutputs()) {
    malloc_buffer(o);
    stack_.declare(CppDType(o->dtype), "output_" + IdxTensor(o),
                   static_cast<size_t>(o->GetSize()->value));
  }
  // read and test datas
  stack_.comment("Read and test datas")
      .while_start("reader.ReadNext(cpu_buffers)")
      .comment("Memcopy inputs host to device");
  // copy inputs
  for (const auto& i : graph()->GetInputs()) {
    stack_.func_call("CHECK")
        .func_call("cudaMemcpyAsync")
        .call_arg(DocUtils::ToIndex("gpu_buffers", "idx_" + IdxTensor(i)))
        .call_arg(DocUtils::ToIndex("cpu_buffers", "idx_" + IdxTensor(i)))
        .call_arg(GetTensorBytes(i))
        .call_arg("cudaMemcpyHostToDevice")
        .call_arg("stream")
        .pop_nest();
  }
  // enqueue
  stack_.func_call("cudaStreamSynchronize")
      .call_arg("stream")
      .comment("enquque with gpu buffers")
      .func_call("enqueueV2", NullOpt, DocUtils::ToPtr("context"))
      .call_arg("gpu_buffers")
      .call_arg("stream")
      .call_arg("nullptr")
      .comment("Memcopy outputs device to host");
  // copy outputs
  for (const auto& o : graph()->GetOutputs()) {
    stack_.func_call("CHECK")
        .func_call("cudaMemcpyAsync")
        .call_arg("output_" + IdxTensor(o))
        .call_arg(DocUtils::ToIndex("gpu_buffers", "idx_" + IdxTensor(o)))
        .call_arg(GetTensorBytes(o))
        .call_arg("cudaMemcpyDeviceToHost")
        .call_arg("stream")
        .pop_nest();
  }
  stack_.func_call("cudaStreamSynchronize").call_arg("stream");
  // compare outputs
  for (const auto& o : graph()->GetOutputs()) {
    stack_.func_call("CommonUtils::CompareBuffers", "pass")
        .call_arg("(" + CppDType(o->dtype) + "*)cpu_buffers[idx_" + IdxTensor(o) + "]")
        .call_arg("output_" + IdxTensor(o))
        .call_arg(o->GetSize());
    ReturnOnFail("pass", "Failed to test the output " + o->name);
  }
  stack_.while_end();
  // clean up
  stack_.comment("Clean up the buffers and stream")
      .func_call("cudaStreamDestroy")
      .call_arg("stream")
      .for_start("i", 0, binding_num)
      .func_call("CHECK")
      .func_call("cudaFree")
      .call_arg(DocUtils::ToIndex("gpu_buffers", "i"))
      .pop_nest()
      .func_call("free")
      .call_arg(DocUtils::ToIndex("cpu_buffers", "i"))
      .for_end();
  // end define test method
  stack_.func_end("true");
  EndNamespace();
}

void TensorRTCodeGen::CodeGenMain() {
  stack_.line("#include \"" + graph()->name + ".h\"")
      .line()
      .line("using namespace nvinfer1;")
      .line("using namespace tvm::contrib::msc;")
      .line()
      .func_def("main", "int")
      .func_arg("argc", "int")
      .func_arg("argv", "char**")
      .func_start()
      .declare("TRTLogger", "logger")
      .func_call("setLogSeverity", "", "logger");
  if (config()->log_level == 0) {
    stack_.call_arg("ILogger::Severity::kINFO");
  } else if (config()->log_level == 1) {
    stack_.call_arg("ILogger::Severity::kVERBOSE");
  } else {
    stack_.call_arg("ILogger::Severity::kWARNING");
  }
  // prepare for build
  stack_.comment("Define arguments")
      .assign("pass", "true", "bool")
      .assign("repeat_num", "1000", "int")
      .assign("profile_level", std::to_string(config()->profile_level), "int")
      .cond_if("argc > 1")
      .assign("profile_level", "atoi(argv[1])")
      .cond_end();

  // start build the engine
  stack_.comment("Build engine if not exist")
      .cond_if("!FileUtils::FileExist(\"" + graph()->name + ".trt\")");
  // create builder
  stack_.comment("Create TensorRT tools")
      .func_call("TRTPtr<IBuilder>", DocUtils::ToDeclare("auto", "builder"))
      .func_call("createInferBuilder")
      .call_arg("logger")
      .pop_nest();
  ReturnOnFail("builder", "Failed to create builder");
  // create network
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_
        .assign("flags",
                "1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)",
                "uint32_t")
        .func_call("TRTPtr<INetworkDefinition>", DocUtils::ToDeclare("auto", "network"))
        .func_call("createNetworkV2", NullOpt, DocUtils::ToPtr("builder"))
        .call_arg("flags")
        .pop_nest();
  } else {
    stack_.func_call("TRTPtr<INetworkDefinition>", DocUtils::ToDeclare("auto", "network"))
        .func_call("createNetwork", NullOpt, DocUtils::ToPtr("builder"))
        .pop_nest();
  }
  ReturnOnFail("network", "Failed to create network");
  // create config
  stack_.func_call("TRTPtr<IBuilderConfig>", DocUtils::ToDeclare("auto", "config"))
      .func_call("createBuilderConfig", NullOpt, DocUtils::ToPtr("builder"))
      .pop_nest();
  ReturnOnFail("config", "Failed to create config");
  // add codegen before build
  for (const auto& l : before_build_codes_) {
    stack_.line(l);
  }
  // build model
  stack_.comment("Build model")
      .declare(graph()->name, "model")
      .func_call("Build", "pass", "model")
      .call_arg("builder")
      .call_arg("network");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.call_arg("config");
  }
  stack_.call_arg("logger");
  ReturnOnFail("pass", "Failed to build model");
  // add codegen after build
  for (const auto& l : after_build_codes_) {
    stack_.line(l);
  }
  // Set profile flag
  stack_.comment("Set profile flag")
      .declare("ProfilingVerbosity", "profile_verbose")
      .cond_if("profile_level == 2")
      .assign("profile_verbose", "ProfilingVerbosity::kDETAILED")
      .cond_else()
      .cond_if("profile_level == 1")
      .assign("profile_verbose", "ProfilingVerbosity::kLAYER_NAMES_ONLY")
      .cond_else()
      .assign("profile_verbose", "ProfilingVerbosity::kNONE")
      .cond_end()
      .cond_end()
      .func_call("setProfilingVerbosity", NullOpt, DocUtils::ToPtr("config"))
      .call_arg("profile_verbose");
  // Serialize engine
  stack_.comment("Serialize engine")
      .func_call("TRTUtils::SerializeEngineToFile", "pass")
      .call_arg(DocUtils::ToStr(graph()->name + ".trt"))
      .call_arg("builder")
      .call_arg("network");
  if (CompareVersion(6, 0, 0) >= 0) {
    stack_.call_arg("config");
  }
  stack_.call_arg("logger");
  ReturnOnFail("pass", "Failed to serialize the engine");
  // end build the engine
  stack_.cond_end();
  // start deserialize engine
  stack_.comment("Deserialize engine")
      .declare("std::shared_ptr<ICudaEngine>", "engine")
      .func_call("TRTUtils::DeserializeEngineFromFile", "pass")
      .call_arg(DocUtils::ToStr(graph()->name + ".trt"))
      .call_arg("engine")
      .call_arg("logger");
  ReturnOnFail("pass", "Failed to deserialize the engine");
  // dump info by inspector
  stack_.comment("Dump info by inspector")
      .cond_if("profile_level > 0")
      .func_call("TRTPtr<IEngineInspector>", DocUtils::ToDeclare("auto", "inspector"))
      .func_call("createEngineInspector", NullOpt, DocUtils::ToPtr("engine"))
      .pop_nest()
      .func_call("getEngineInformation", DocUtils::ToDeclare("std::string", "result"),
                 DocUtils::ToPtr("inspector"))
      .call_arg("LayerInformationFormat::kJSON")
      .declare("std::ofstream", "os")
      .declare_arg(DocUtils::ToStr(graph()->name + "_info.json"))
      .declare_arg("std::ofstream::trunc")
      .line("os << result << std::flush;")
      .cond_end();
  // test engine
  if (config()->test_iter > 0) {
    stack_.comment("Prepare dataset")
        .declare("DatasetReader", "reader")
        .declare_arg(DocUtils::ToStr(config()->dataset))
        .declare_arg(config()->test_iter);
    stack_.comment("Test engine by datas")
        .func_call("test_" + graph()->name, "pass")
        .call_arg("engine")
        .call_arg("reader")
        .call_arg("logger");
  }
  ReturnOnFail("pass", "Failed to test the engine");
  stack_.func_end("pass ? 0 : 1");
}

void TensorRTCodeGen::CodeGenCmake() {
  stack_.line("cmake_minimum_required(VERSION " + config()->cmake_version + " FATAL_ERROR)")
      .line("project(" + graph()->name + ")")
      .line("find_package(CUDA)")
      .line()
      .line("find_path(TRT_INCLUDE_DIR NvInfer.h HINTS " + config()->tensorrt_root +
            " PATH_SUFFIXES include)")
      .line("find_library(TRT_LIBS nvinfer HINTS " + config()->tensorrt_root +
            " PATH_SUFFIXES lib)")
      .line(
          "message(STATUS \"Build project with TRT_INCLUDE_DIR ${TRT_INCLUDE_DIR} and "
          "TRT_LIBS "
          "${TRT_LIBS}\")")
      .line()
      .line("add_definitions(-DTRT_MAJOR=" + std::to_string(config()->version[0]) + ")")
      .line("add_definitions(-DTRT_MINOR=" + std::to_string(config()->version[1]) + ")")
      .line("add_definitions(-DTRT_PATCH=" + std::to_string(config()->version[2]) + ")")
      .line();
  if (config()->use_plugin) {
    stack_.line("add_definitions(-DPLUGIN_SUPPORT_TENSORRT)").line();
  }
  String link_libs = " ${TRT_LIBS}";
  if (config()->extern_libs.size() > 0) {
    stack_.line("set(EXTERN_LIBS " + StringUtils::Join(config()->extern_libs, " ") + ")");
    link_libs = link_libs + " ${EXTERN_LIBS}";
  }
  stack_.line("file(GLOB_RECURSE TRT_SRCS *.cc)")
      .line("cuda_add_executable(" + graph()->name + " ${TRT_SRCS})")
      .line("target_include_directories(" + graph()->name + " PUBLIC ${TRT_INCLUDE_DIR})")
      .line("target_link_libraries(" + graph()->name + link_libs + ")");
}

const String TensorRTCodeGen::IdxTensor(const MSCTensor& tensor) {
  const auto& pair = graph()->FindProducerAndIdx(tensor);
  const String& prefix = "tensor_" + std::to_string(pair.first->index);
  if (pair.first->outputs.size() > 1) {
    return prefix + "_" + std::to_string(pair.second);
  }
  return prefix;
}

const String TensorRTCodeGen::CppDType(const DataType& dtype) {
  const String& dtype_name = CppCodeGen<TensorRTCodeGenConfig, TensorRTCodeGenHelper>::DType(dtype);
  if (dtype_name == "int32") {
    return "int";
  }
  if (dtype_name == "int64") {
    return "int64_t";
  }
  if (dtype_name == "float32") {
    return "float";
  }
  if (dtype_name == "float64") {
    return "double";
  }
  return dtype_name;
}

const String TensorRTCodeGen::GetTensorBytes(const MSCTensor& tensor) {
  return std::to_string(tensor->GetSize()->value) + " * sizeof(" + CppDType(tensor->dtype) + ")";
}

void TensorRTCodeGen::ReturnOnFail(const String& flag, const String& err) {
  stack_.cond_if("!" + flag)
      .func_call("logger.log")
      .call_arg("ILogger::Severity::kERROR")
      .call_arg(DocUtils::ToStr(err))
      .line("return -1;")
      .cond_end();
}

template <typename T>
const String TensorRTCodeGen::ToDims(const std::vector<T>& dims, bool use_ndim) {
  if (dims.size() == 2 && !use_ndim) {
    return "DimsHW{" + std::to_string(dims[0]) + "," + std::to_string(dims[1]) + "}";
  }
  String dims_str = "Dims({" + std::to_string(dims.size()) + ",{";
  for (size_t i = 0; i < dims.size(); i++) {
    dims_str = dims_str + std::to_string(dims[i]) + (i < dims.size() - 1 ? "," : "");
  }
  dims_str = dims_str + "}})";
  return dims_str;
}

const String TensorRTCodeGen::ToDims(const Array<Integer>& dims, bool use_ndim) {
  std::vector<int64_t> int_dims;
  for (const auto& d : dims) {
    int_dims.push_back(d->value);
  }
  return ToDims(int_dims, use_ndim);
}

const Array<Doc> TensorRTCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetTensorRTOpCodes();
  auto it = ops_map->find(GetOpType(node));
  ICHECK(it != ops_map->end()) << "Unsupported tensorrt op(" << node->optype << "): " << node;
  it->second->Config(node, config());
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

const Map<String, String> TensorRTCodeGen::GetTensorCtx(const MSCTensor& tensor) {
  Map<String, String> tensor_ctx;
  tensor_ctx.Set("ctx", "network");
  for (const auto& pair :
       CppCodeGen<TensorRTCodeGenConfig, TensorRTCodeGenHelper>::GetTensorCtx(tensor)) {
    tensor_ctx.Set(pair.first, pair.second);
  }
  return tensor_ctx;
}

const Map<String, String> TensorRTCodeGen::GetStepCtx() {
  Map<String, String> step_ctx;
  step_ctx.Set("network", "network");
  step_ctx.Set("config", "config");
  step_ctx.Set("builder", "builder");
  for (const auto& pair : CppCodeGen<TensorRTCodeGenConfig, TensorRTCodeGenHelper>::GetStepCtx()) {
    step_ctx.Set(pair.first, pair.second);
  }
  return step_ctx;
}

TVM_REGISTER_GLOBAL("msc.framework.tensorrt.GetTensorRTSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String& print_config) -> Map<String, String> {
      TensorRTCodeGen codegen = TensorRTCodeGen(graph, codegen_config);
      return codegen.GetSources(print_config);
    });

TVM_REGISTER_GLOBAL("msc.framework.tensorrt.GetTensorRTRoot").set_body_typed([]() -> String {
#ifdef TENSORRT_ROOT_DIR
  return TENSORRT_ROOT_DIR;
#else
  return "";
#endif
});

/*!
 * \brief Create runtime modules for MSC TensorRT.
 * \param functions The extern functions to be compiled via TensorRT
 * \return Runtime modules.
 */
Array<runtime::Module> MSCTensorRTCompiler(Array<Function> functions,
                                           Map<String, ObjectRef> target_option,
                                           Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;
  for (const auto& func : functions) {
    VLOG(1) << "MSC.TensorRT partition:" << std::endl << func;
    const auto& name_opt = func->GetAttr<runtime::String>(msc_attr::kUnique);
    ICHECK(name_opt.defined()) << "Can not find " << msc_attr::kUnique << " from attrs";
    const auto& name = name_opt.value();
    std::string func_name = GetExtSymbol(func);
    ICHECK(target_option.count(name)) << "Can not find target option for " << name;
    const auto& options = Downcast<String>(target_option[name]);
    MSCJSONSerializer serializer(constant_names, options);
    serializer.serialize(func);
    std::string graph_json = serializer.GetJSON();
    const auto* pf = runtime::Registry::Get("runtime.msc_tensorrt_runtime_create");
    ICHECK(pf != nullptr) << "Cannot find TensorRT runtime module create function.";
    VLOG(1) << "Creating msc_tensorrt runtime::Module for '" << func_name << "'";
    compiled_functions.push_back((*pf)(func_name, graph_json, serializer.GetConstantNames()));
  }
  return compiled_functions;
}

TVM_REGISTER_GLOBAL("relax.ext.msc_tensorrt").set_body_typed(MSCTensorRTCompiler);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
