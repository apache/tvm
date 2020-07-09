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
 * \file src/runtime/contrib/acl/acl_kernel.h
 * \brief Use ACL library kernels, we create an interface to these.
 */

#ifndef TVM_RUNTIME_CONTRIB_ACL_ACL_KERNEL_H_
#define TVM_RUNTIME_CONTRIB_ACL_ACL_KERNEL_H_

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/Tensor.h>
#include <dmlc/filesystem.h>
#include <dmlc/json.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>

#include <memory>
#include <utility>
#include <vector>

#include "../../../relay/backend/contrib/acl/acl_api.h"
#include "acl_allocator.h"
#include "acl_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {
namespace acl {

namespace api = relay::contrib::acl;
namespace acl = arm_compute;

/*!
 * \brief ACL objects we cache in order to avoid needing to construct
 * a new layer each time.
 */
struct CacheItems {
  std::shared_ptr<arm_compute::IFunction> function;
  std::vector<arm_compute::Tensor> inputs;
  std::vector<arm_compute::Tensor> const_inputs;
  std::vector<arm_compute::Tensor> outputs;
};

/*!
 * \brief A cached ACL layer containing a single ACL function.
 */
class CachedLayer {
 public:
  /*!
   * \brief Create an ACL layer from a JSON representation. Also prepare
   * the layer for execution - this will perform actions such as pre-
   * transposing of weights.
   *
   * \note The naming suggests a subgraph directly maps to a layer.
   * In general this is not true, but since ACL only expects subgraphs
   * consisting of a single op it is.
   *
   * \param function A JSON representation of a subgraph.
   * \param constants The constants used in the subgraph.
   * \param allocator ACL can request memory from TVM.
   */
  CachedLayer(const api::JSONSubGraph& function, const std::vector<NDArray>& constants,
              ACLAllocator* allocator, const std::shared_ptr<acl::MemoryManagerOnDemand>& mm);

  /*!
   * \brief Run inference on the ACL layer.
   *
   * \param inputs The inputs for the layer.
   * \param outputs The outputs for the layer.
   * \return True if success, False if not successful.
   */
  bool Inference(const std::vector<DLTensor*>& inputs, const std::vector<DLTensor*>& outputs);

  /*!
   * \brief Get the number of inputs the layer takes.
   *
   * \return Number of inputs.
   */
  size_t GetNumInputs() const;

  /*!
   * \brief Check if the layer requires working memory to be allocated.
   *
   * \return True if it does, False if not.
   */
  bool IsMemoryManaged() const { return this->is_mm_; }

 private:
  /*! \brief Constant tensors used in the layer. */
  std::vector<NDArray> constants_;
  /*! \brief Cache ACL function and tensors for execution. */
  CacheItems function_;
  /*! \brief ACL Allocator to request auxiliary memory from TVM. */
  ACLAllocator* allocator_;
  /*! \brief Check if the function requires working memory to be allocated. */
  bool is_mm_ = false;

  /*! \brief Create individual ACL layer. */
  static void CreateConvolution2DLayer(CacheItems* cache, const api::JSONOp& params,
                                       const std::shared_ptr<acl::MemoryManagerOnDemand>& mm);
  static void CreateMaxPoolLayer(CacheItems* cache, const api::JSONOp& params);
  static void CreateReshapeLayer(CacheItems* cache, const api::JSONOp& params);
};

}  // namespace acl
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_ACL_ACL_KERNEL_H_
