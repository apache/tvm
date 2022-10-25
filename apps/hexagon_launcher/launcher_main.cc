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

#include <getopt.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

#include "launcher_core.h"
#include "launcher_util.h"

ExecutionSession* create_execution_session(bool gen_lwp_json);

int parse_command_line(int argc, char* argv[], std::string* in_path, std::string* out_path,
                       bool* gen_lwp_json) {
  static option long_options[] = {
      {"in_config", required_argument, nullptr, 0},
      {"out_config", required_argument, nullptr, 0},
      {"gen_lwp_json", optional_argument, nullptr, 0},
  };

  bool show_usage = false;
  int opt, long_index = 0;
  while ((opt = getopt_long(argc, argv, "i:o:u:", long_options, &long_index)) != -1) {
    if (opt != 0) {
      show_usage = true;
      continue;
    }
    switch (long_index) {
      case 0:
        *in_path = std::string(optarg);
        break;
      case 1:
        *out_path = std::string(optarg);
        break;
      case 2:
        *gen_lwp_json = true;
        break;
    }
  }
  if (in_path->empty() || out_path->empty() || show_usage) {
    std::cout << "Usage: " << argv[0] << " --" << long_options[0].name << " input.json --"
              << long_options[1].name << " output.json\n";
    return 1;
  }
  return 0;
}

int main(int argc, char* argv[]) {
  std::string in_path, out_path;
  bool gen_lwp_json;
  if (parse_command_line(argc, argv, &in_path, &out_path, &gen_lwp_json) != 0) {
    return 1;
  }

  ModelConfig config;
  if (!read_model_config(in_path, &config)) {
    return 1;
  }

  ExecutionSession* session_ptr = create_execution_session(gen_lwp_json);
  if (session_ptr == nullptr) {
    return 1;
  }
  ExecutionSession& session = *session_ptr;

  std::cout << "loading model files: ";
  if (!config.model_json.empty()) {
    std::cout << config.model_json << ", ";
  }
  std::cout << config.model_library << '\n';

  std::string json = !config.model_json.empty() ? load_text_file(config.model_json) : "";
  if (!session.load_model(config.model_library, json.c_str())) {
    return 1;
  }

  int max_ndim = 0;
  for (const TensorConfig& tc : config.inputs) {
    max_ndim = std::max<int>(max_ndim, tc.shape.size());
  }
  auto* input_meta = session.alloc<tensor_meta>(tensor_meta::meta_size(max_ndim));

  for (int i = 0, e = config.inputs.size(); i != e; ++i) {
    const TensorConfig& tc = config.inputs[i];
    input_meta->ndim = tc.shape.size();
    input_meta->dtype = tvm::runtime::String2DLDataType(tc.dtype);
    std::copy(tc.shape.begin(), tc.shape.end(), input_meta->shape);

    auto* input_data = session.alloc<unsigned char>(input_meta->data_size());
    std::cout << "loading input file #" << i << ": " << tc.file_name << '\n';
    load_binary_file(tc.file_name, input_data, input_meta->data_size());
    if (!session.set_input(i, input_meta, input_data)) {
      return 1;
    }
  }

  OutputConfig output_config;

  std::cout << "running..." << std::flush;
  if (!session.run(&output_config.pcycles, &output_config.usecs)) {
    std::cout << '\n';
    return 1;
  }
  std::cout << '\n';
  std::cout << "Finished in " << output_config.pcycles << " pcycles, (" << output_config.usecs
            << "us)\n";

  auto* output_meta = session.alloc<tensor_meta>(128);
  int num_outputs = 0;
  if (!session.get_num_outputs(&num_outputs)) {
    return 1;
  }

  for (int i = 0; i != num_outputs; ++i) {
    if (!session.get_output(i, output_meta, 128, nullptr, 0)) {
      return 1;
    }
    int data_size = output_meta->data_size();
    auto* output_data = session.alloc<unsigned char>(data_size);
    if (!session.get_output(i, output_meta, 128, output_data, data_size)) {
      return 1;
    }

    TensorConfig oc;
    oc.file_name = "output" + std::to_string(i) + ".dat";
    for (int i = 0, e = output_meta->ndim; i != e; ++i) {
      oc.shape.push_back(output_meta->shape[i]);
    }
    oc.dtype = tvm::runtime::DLDataType2String(output_meta->dtype);
    write_binary_file(oc.file_name, output_data, data_size);
    output_config.outputs.push_back(std::move(oc));

    session.free(output_data);
  }

  if (!write_output_config(out_path, &output_config)) {
    return 1;
  }
  return 0;
}
