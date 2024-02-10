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
#ifndef TVM_RUNTIME_RELAX_VM_RNN_STATE_H_
#define TVM_RUNTIME_RELAX_VM_RNN_STATE_H_
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstdint>

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief The base class of RNN State for efficient
 * State data management and attention computation.
 */
class RNNState : public Object {
 public:
  /*! \brief Reset the RNN State. */
  virtual void Clear() = 0;

  /************** Sequence Management **************/

  /*!
   * \brief Add a new sequence with init State data in the cache.
   * Check if the validity of the input sequence id.
   * \param seq_id The id of the new sequence to be added.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void AddSequence(int64_t seq_id) = 0;

  /*!
   * \brief Remove a sequence and its State data from the KV cache.
   * \param seq_id The sequence to remove from cache.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void RemoveSequence(int64_t seq_id) = 0;

  /*!
   * \brief Fork the State data of parent sequence to the child sequence.
   * After the fork, the child sequence has State data of the parent
   * sequence.
   * \param parent_seq_id The parent (source) of the fork.
   * \param child_seq_id The child (destination) of the fork.
   * The child sequence id should not exist in cache prior to fork.
   * \throws Error if the given sequence ids are not valid.
   */
  virtual void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id) = 0;

  /*!
   * \brief Rollback the State data of the specified sequence.
   * \param seq_id The sequence whose State data is to be rolled back.
   * \param n The number of tokens to be rolled back.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void PopN(int64_t seq_id, int64_t n) = 0;

  /************** Interaction **************/

  /*!
   * \brief Mark the start of the forward function with the ids of
   * the sequences and the sequence length to forward for each
   * sequence.
   * \param seq_id The sequence whose forward pass is to be started.
   * \param append_lengths The length of the sequence to forward.
   */
  virtual void BeginForward(const IntTuple& seq_ids, const IntTuple& append_lengths) = 0;

  /*!
   * \brief Mark the start of the forward function.
   * This method is invoked right after entering the model forward
   * function, and contains post-processing of the forward.
   */
  virtual void EndForward() = 0;

  /*!
   * \brief Get the State data for the specified sequence.
   * \param layer_id The model layer where the state is set.
   * \param state_id The state id within the layer.
   * \return The array of State data, each element corresponds to a state.
   * \throws Error if the given sequence id is not valid.
   */
  virtual NDArray Get(int64_t layer_id, int64_t state_id) = 0;

  /*!
   * \brief Set the State data for the specified sequence.
   * \param layer_id The model layer where the state is set.
   * \param state_id The state id within the layer.
   * \param data The data to be set.
   * \throws Error if the given sequence id is not valid.
   */
  virtual void Set(int64_t layer_id, int64_t state_id, NDArray data) = 0;

  /************** Compatibility with KVCache **************/
};
}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RELAX_VM_RNN_STATE_H_
