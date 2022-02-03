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
 * \file src/contrib/ethosu/cascader/cascader_options.h
 * \brief Class to store configuration options for the NPU cascader
 */
#ifndef TVM_CONTRIB_ETHOSU_CASCADER_CASCADER_OPTIONS_H_
#define TVM_CONTRIB_ETHOSU_CASCADER_CASCADER_OPTIONS_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include "tensor_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

/*! \brief Node to represent CascaderOptions */
class CascaderOptionsNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v);

  /*! \brief The MemoryRegion to place cascading buffer into. */
  MemoryRegion cascade_region;
  /*! \brief The maximum number of Proposals to generate. */
  int max_proposals;
  /*! \brief How many striping factors to try per axis. */
  int stripe_factors;
  /*! \brief The maximum number of Parts in a Plan. */
  int max_plan_size;
  /*! \brief The maximum size of Tensor that will always be copied into the cascade region. */
  int always_copy_size;

  static constexpr const char* _type_key = "contrib.ethosu.cascader.CascaderOptions";
  TVM_DECLARE_FINAL_OBJECT_INFO(CascaderOptionsNode, Object)
};

/*! \brief A class to hold configuration options for the cascader. */
class CascaderOptions : public ObjectRef {
 public:
  CascaderOptions(const MemoryRegion& cascade_region, int max_proposals, int stripe_factors,
                  int max_plan_size, int always_copy_size);

  TVM_DEFINE_OBJECT_REF_METHODS(CascaderOptions, ObjectRef, CascaderOptionsNode);
};

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_ETHOSU_CASCADER_CASCADER_OPTIONS_H_
