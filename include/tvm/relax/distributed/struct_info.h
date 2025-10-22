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
 * \file tvm/relax/distributed/struct_info.h
 * \brief Struct info for DTensor (Distributed Tensor)
 */

#ifndef TVM_RELAX_DISTRIBUTED_STRUCT_INFO_H_
#define TVM_RELAX_DISTRIBUTED_STRUCT_INFO_H_

#include <tvm/relax/distributed/global_info.h>
#include <tvm/relax/struct_info.h>
namespace tvm {
namespace relax {
namespace distributed {

enum class PlacementSpecKind : int { kSharding = 0, kReplica = 1 };

/*! \brief Describes how data is distributed in one dimension of the device mesh*/
class PlacementSpecNode : public Object {
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
  TVM_FFI_DECLARE_OBJECT_INFO("relax.distributed.PlacementSpec", PlacementSpecNode, Object);
};

/*!
 * \brief Managed reference to PlacementSpecNode.
 * \sa PlacementSpecNode
 */
class PlacementSpec : public ObjectRef {
 public:
  TVM_DLL static PlacementSpec Sharding(int axis);

  TVM_DLL static PlacementSpec Replica();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PlacementSpec, ObjectRef, PlacementSpecNode);
};

class ShardingNode : public PlacementSpecNode {
 public:
  /*! \brief The dimension of tensor we shard*/
  Integer sharding_dim;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ShardingNode>().def_ro("sharding_dim", &ShardingNode::sharding_dim);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.distributed.Sharding", ShardingNode, PlacementSpecNode);
};

/*! \brief Describes how data is distributed in each dimension of the device mesh*/
class PlacementNode : public Object {
 public:
  /*! \brief specs for each dim of device mesh.*/
  ffi::Array<PlacementSpec> dim_specs;

  ffi::String ToString() const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PlacementNode>().def_ro("dim_specs", &PlacementNode::dim_specs);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindConstTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.distributed.Placement", PlacementNode, Object);
};

/*!
 * \brief Managed reference to a Placement.
 * \sa PlacementNode
 */
class Placement : public ObjectRef {
 public:
  TVM_DLL explicit Placement(ffi::Array<PlacementSpec> dim_specs);
  /*! \brief replica dim is printed as "R" and sharding dim is printed as "S[i]".]*/
  static Placement FromText(ffi::String text_repr);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Placement, ObjectRef, PlacementNode);
};

/*!
 * \brief StructInfo of DTensor (Distributed Tensor).
 */
class DTensorStructInfoNode : public StructInfoNode {
 public:
  /*!
   * \brief The struct info inherited from TensorStructInfo
   */
  TensorStructInfo tensor_sinfo;
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
    refl::ObjectDef<DTensorStructInfoNode>()
        .def_ro("device_mesh", &DTensorStructInfoNode::device_mesh)
        .def_ro("placement", &DTensorStructInfoNode::placement)
        .def_ro("tensor_sinfo", &DTensorStructInfoNode::tensor_sinfo);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.DTensorStructInfo", DTensorStructInfoNode,
                                    StructInfoNode);
};

/*!
 * \brief Managed reference to DTensorStructInfoNode.
 * \sa DTensorStructInfoNode
 */
class DTensorStructInfo : public StructInfo {
 public:
  /*!
   * \brief Construction with device mesh and placement.
   * \param tensor_sinfo The struct info inherited from TensorStructInfo
   * \param device_mesh The device mesh of the tensor.
   * \param placement The placement of the tensor among the device mesh.
   * \param span The span of the AST.
   */
  TVM_DLL DTensorStructInfo(TensorStructInfo tensor_sinfo, DeviceMesh device_mesh,
                            Placement placement, Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DTensorStructInfo, StructInfo, DTensorStructInfoNode);
};

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_STRUCT_INFO_H_
