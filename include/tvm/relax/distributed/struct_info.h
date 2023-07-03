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

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("axis", &axis);
    v->Visit("kind", &kind);
  }
  bool SEqualReduce(const PlacementSpecNode* other, SEqualReducer equal) const {
    return equal(axis, other->axis) && equal(kind, other->kind);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(axis);
    hash_reduce(static_cast<int>(kind));
  }

  static constexpr const char* _type_key = "relax.distributed.PlacementSpec";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(PlacementSpecNode, Object);
};

/*!
 * \brief Managed reference to PlacementSpecNode.
 * \sa PlacementSpecNode
 */
class PlacementSpec : public ObjectRef {
 public:
  TVM_DLL static PlacementSpec Sharding(int axis);

  TVM_DLL static PlacementSpec Replica();

  TVM_DEFINE_OBJECT_REF_METHODS(PlacementSpec, ObjectRef, PlacementSpecNode);
};

class ShardingNode : public PlacementSpecNode {
 public:
  /*! \brief The dimension of tensor we shard*/
  Integer sharding_dim;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("sharding_dim", &sharding_dim); }

  bool SEqualReduce(const ShardingNode* other, SEqualReducer equal) const {
    return equal(sharding_dim, other->sharding_dim);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(sharding_dim); }
  static constexpr const char* _type_key = "relax.distributed.Sharding";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShardingNode, PlacementSpecNode);
};

/*! \brief Describes how data is distributed in each dimension of the device mesh*/
class PlacementNode : public Object {
 public:
  /*! \brief specs for each dim of device mesh.*/
  Array<PlacementSpec> dim_specs;

  String ToString() const;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("dim_specs", &dim_specs); }

  bool SEqualReduce(const PlacementNode* other, SEqualReducer equal) const {
    return equal(dim_specs, other->dim_specs);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(dim_specs); }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "relax.distributed.Placement";
  TVM_DECLARE_FINAL_OBJECT_INFO(PlacementNode, Object);
};

/*!
 * \brief Managed reference to a Placement.
 * \sa PlacementNode
 */
class Placement : public ObjectRef {
 public:
  TVM_DLL explicit Placement(Array<PlacementSpec> dim_specs);
  /*! \brief replica dim is printed as "R" and sharding dim is printed as "S[i]".]*/
  static Placement FromText(String text_repr);
  TVM_DEFINE_OBJECT_REF_METHODS(Placement, ObjectRef, PlacementNode);
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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("device_mesh", &device_mesh);
    v->Visit("placement", &placement);
    v->Visit("tensor_sinfo", &tensor_sinfo);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const DTensorStructInfoNode* other, SEqualReducer equal) const {
    return equal(tensor_sinfo, other->tensor_sinfo) && equal(device_mesh, other->device_mesh) &&
           equal(placement, other->placement);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tensor_sinfo);
    hash_reduce(device_mesh);
    hash_reduce(placement);
  }

  static constexpr const char* _type_key = "relax.DTensorStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(DTensorStructInfoNode, StructInfoNode);
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

  TVM_DEFINE_OBJECT_REF_METHODS(DTensorStructInfo, StructInfo, DTensorStructInfoNode);
};

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_STRUCT_INFO_H_
