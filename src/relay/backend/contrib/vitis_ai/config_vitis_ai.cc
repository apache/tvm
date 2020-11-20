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
 * \file src/relay/backend/contrib/vitis_ai/config_vitis_ai.cc
 * \brief Register Vitis-AI codegen options. Main codegen is implemented in python.
 */

#include <tvm/ir/transform.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace vitis_ai {

/*! \brief The target Vitis-AI accelerator device */
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.vitis_ai.options.target", String);
/*! \brief (Optional config) The build directory to be used by Vitis-AI */
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.vitis_ai.options.build_dir", String);
/*! \brief (Optional config) The work directory to be used by Vitis-AI */
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.vitis_ai.options.work_dir", String);
/*! \brief (Optional config) Export PyXIR runtime module to disk during serialization if provided */
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.vitis_ai.options.export_runtime_module", String);
/*! \brief (Optional config) Load PyXIR runtime module from disk */
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.vitis_ai.options.load_runtime_module", String);

}  // namespace vitis_ai
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
