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

// LINT_C_FILE

/*!
 * \file tvm/runtime/metadata_types.h
 * \brief Defines types which can be used in metadata here which
 * are also shared between C and C++ code bases.
 */
#ifndef TVM_RUNTIME_METADATA_TYPES_H_
#define TVM_RUNTIME_METADATA_TYPES_H_

#include <inttypes.h>
#include <tvm/runtime/c_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Top-level metadata structure. Holds all other metadata types.
 */
struct TVMMetadata {
  /*! \brief Version identifier for this metadata. */
  int64_t version;
  /*! \brief Inputs to the AOT run_model function.
   * The order of the elements is the same as in the arguments to run_model. That is to say,
   * this array specifies the first `num_inputs` arguments to run_model.
   */
  const struct TVMTensorInfo* inputs;
  /*! \brief Number of elements in `inputs` array. */
  int64_t num_inputs;
  /*! \brief Outputs of the AOT run_model function.
   * The order of the elements is the same as in the arguments to run_model. That is to say,
   * this array specifies the last `num_outputs` arguments to run_model.
   */
  const struct TVMTensorInfo* outputs;
  /*! \brief Number of elements in `outputs` array. */
  int64_t num_outputs;
  /*! \brief Workspace Memory Pools needed by the AOT main function.
   * The order of the elements is the same as in the arguments to run_model. That is to say,
   * this array specifies the last `num_workspace_pools` arguments to run_model.
   */
  const struct TVMTensorInfo* workspace_pools;
  /*! \brief Number of elements in `workspace_pools` array. */
  int64_t num_workspace_pools;
  /*! \brief Constant pools needed by the AOT main function.
   */
  const struct TVMConstantInfo* constant_pools;
  /*! \brief Number of elements in `constant_pools` array. */
  int64_t num_constant_pools;
  /*! \brief Name of the model, as passed to tvm.relay.build. */
  const char* mod_name;
};

/*!
 * \brief Describes one tensor argument to `run_model`.
 * NOTE: while TIR allows for other types of arguments, such as scalars, the AOT run_model
 * function does not currently accept these. Therefore it's not possible to express those
 * in this metadata. A future patch may modify this.
 */
struct TVMTensorInfo {
  /*! \brief Name of the tensor, as specified in the Relay program. */
  const char* name;
  /*! \brief Shape of the tensor. */
  const int64_t* shape;
  /*! \brief Rank of this tensor. */
  int64_t num_shape;
  /*! \brief Data type of one element of this tensor. */
  DLDataType dtype;
};

/*!
 * \brief Describes one constant argument to `run_model`.
 *
 */
struct TVMConstantInfo {
  /*! \brief Name of the constant */
  const char* name_hint;
  /*! \brief Offset in bytes of the constant */
  int64_t byte_offset;
  /*! \brief length of the data_bytes field */
  int64_t data_len;
  /*! \brief data bytes of serialized NDArray */
  const void* data_bytes;
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_RUNTIME_METADATA_TYPES_H_
