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
 * \file tvm/relax/distributed/type.h
 * \brief Type definitions for DTensor (Distributed Tensor)
 */

#ifndef TVM_RELAX_DISTRIBUTED_TYPE_H_
#define TVM_RELAX_DISTRIBUTED_TYPE_H_

#include <tvm/relax/distributed/global_info.h>
#include <tvm/relax/type.h>

#include <utility>

namespace tvm {
namespace relax {
namespace distributed {

enum class PlacementSpecKind : int { kSharding = 0, kReplica = 1 };

/*! \brief Describes how data is distributed in one dimension of the device mesh*/
class PlacementSpecNode : public ffi::Object {
 public:
  /*! \brief If the kind is sharding, this value represents the tensor dimension to shard.
   *         otherwise, axis is -1.
   */
  int axis;

  /*! \brief The kind of placement spec. Possible values: kSharding and kReplica. */
  PlacementSpecKind kind;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PlacementSpecNode>()
        .def_ro("axis", &PlacementSpecNode::axis)
        .def_ro("kind", &PlacementSpecNode::kind);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindConstTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.distributed.PlacementSpec", PlacementSpecNode, ffi::Object);
};

/*!
 * \brief Managed reference to PlacementSpecNode.
 * \sa PlacementSpecNode
 */
class PlacementSpec : public ffi::ObjectRef {
 public:
  TVM_DLL static PlacementSpec Sharding(int axis);

  TVM_DLL static PlacementSpec Replica();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PlacementSpec, ffi::ObjectRef, PlacementSpecNode);
};

class ShardingNode : public PlacementSpecNode {
 public:
  /*! \brief The dimension of tensor we shard*/
  int64_t sharding_dim;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShardingNode>().def_ro("sharding_dim", &ShardingNode::sharding_dim);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.distributed.Sharding", ShardingNode, PlacementSpecNode);
};

/*! \brief Describes how data is distributed in each dimension of the device mesh*/
class PlacementNode : public ffi::Object {
 public:
  /*! \brief specs for each dim of device mesh.*/
  ffi::Array<PlacementSpec> dim_specs;

  ffi::String ToString() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PlacementNode>().def_ro("dim_specs", &PlacementNode::dim_specs);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindConstTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.distributed.Placement", PlacementNode, ffi::Object);
};

/*!
 * \brief Managed reference to a Placement.
 * \sa PlacementNode
 */
class Placement : public ffi::ObjectRef {
 public:
  TVM_DLL explicit Placement(ffi::Array<PlacementSpec> dim_specs);
  /*! \brief replica dim is printed as "R" and sharding dim is printed as "S[i]".]*/
  static Placement FromText(ffi::String text_repr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Placement, ffi::ObjectRef, PlacementNode);
};

/*!
 * \brief Type of DTensor (Distributed Tensor).
 */
class DTensorTypeNode : public TypeNode {
 public:
  explicit DTensorTypeNode(ffi::UnsafeInit)
      : tensor_ty(ffi::UnsafeInit{}), device_mesh(), placement() {}

  DTensorTypeNode(TensorType tensor_ty, DeviceMesh device_mesh, Placement placement)
      : tensor_ty(std::move(tensor_ty)),
        device_mesh(std::move(device_mesh)),
        placement(std::move(placement)) {}

  /*!
   * \brief The tensor type carried by the DTensor type.
   */
  TensorType tensor_ty;
  /*!
   * \brief The device mesh of the tensor.
   */
  DeviceMesh device_mesh;
  /*!
   * \brief The placement of the tensor among the device mesh.
   */
  Placement placement;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DTensorTypeNode>()
        .def_ro("device_mesh", &DTensorTypeNode::device_mesh)
        .def_ro("placement", &DTensorTypeNode::placement)
        .def_ro("tensor_ty", &DTensorTypeNode::tensor_ty);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.DTensorType", DTensorTypeNode, TypeNode);
};

/*!
 * \brief Managed reference to DTensorTypeNode.
 * \sa DTensorTypeNode
 */
class DTensorType : public Type {
 public:
  /*!
   * \brief Construction with device mesh and placement.
   * \param tensor_ty The tensor type carried by the DTensor type.
   * \param device_mesh The device mesh of the tensor.
   * \param placement The placement of the tensor among the device mesh.
   * \param span The span of the AST.
   */
  TVM_DLL DTensorType(TensorType tensor_ty, DeviceMesh device_mesh, Placement placement,
                      Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DTensorType, Type, DTensorTypeNode);
};

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_TYPE_H_
