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
 * \file tvm/relax/attrs/distributed.h
 * \brief Attributes for redistribute and annotate_sharding operators.
 */
#ifndef TVM_RELAX_ATTRS_DISTRIBUTED_H_
#define TVM_RELAX_ATTRS_DISTRIBUTED_H_

#include <tvm/relax/distributed/global_info.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes for redistribute and annotate_sharding operator */
struct DistributionAttrs : public tvm::AttrsNode<DistributionAttrs> {
  distributed::DeviceMesh device_mesh;
  distributed::Placement placement;

  TVM_DECLARE_ATTRS(DistributionAttrs, "relax.attrs.DistributionAttrs") {
    TVM_ATTR_FIELD(device_mesh).describe("The device mesh of a tensor's distribution plan");
    TVM_ATTR_FIELD(placement).describe("The placement of a tensor's distribution plan");
  }
};  // struct DistributionAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ATTRS_DISTRIBUTED_H_
