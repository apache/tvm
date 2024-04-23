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
 * \brief Codegen part of tuning capabilities for cublas matmul primitives. 
 */

#include <dmlc/json.h>

#include "../../../../runtime/contrib/cublas/cublas_algo.h"

namespace tvm {
namespace relax {
namespace contrib {

using AlgoCollection = tvm::contrib::AlgoCollection;
using AlgoDesc = tvm::contrib::AlgoDesc;

/*! \brief Algo database with predefined Algo objects. */
class AlgoDatabaseNode: public runtime::Object {
  /*! \brief Mapping of compisite func struct hash to algo colelction. */
  std::map<uint64_t, AlgoCollection> collections;

public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    // v->Visit("collections", &collections);
  }
 
  static constexpr const char* _type_key = "relax.backend.contrib.AlgoDataBase";
  TVM_DECLARE_BASE_OBJECT_INFO(AlgoDatabaseNode, runtime::Object);

  /*! \brief Put new record to database. 
   *
   * \param task_hash Struct hash of corresponding composite function.
   * \param algo_collection The algo collection to register.
   */
  void PutRec(uint64_t task_hash, const AlgoCollection algo_collection);

  /*! \brief Find algo collection for particular func hash. 
   *
   * \param task_hash Struct hash of composite function.
   * \return The AlgoCollection registered for provided hash or empty collection 
   *         if there is no shuch.
   */
  AlgoCollection FindRec(uint64_t task_hash) const;

  /*! \brief Serialize to json form. */ 
  String ToJSON() const;
  
  /*! \brief JSON serialization routine. */
  void Save(dmlc::JSONWriter* writer) const;
  
  /*! \brief JSON deserialization routine. */
  void Load(dmlc::JSONReader* reader);
};

/*!
 * \brief Managed reference to AlgoDatabaseNode
 * \sa AlgoDatabaseNode
 */
class AlgoDatabase: public runtime::ObjectRef {
 public:
  /*! \brief Default constructor. */
  AlgoDatabase();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AlgoDatabase, runtime::ObjectRef, AlgoDatabaseNode);
  
  /*! \brief Operator wrapper on top of FindRec method. */
  AlgoCollection operator()(uint64_t task_hash) const {
    return get()->FindRec(task_hash);
  };
  
  /*! \brief Deserialize from json representation. */ 
  static AlgoDatabase FromJSON(String json);

  /*! \brief Get last databse registered for thread local context. */ 
  static Optional<AlgoDatabase> Current();

  /*! \brief Submit databse to current thread local context. */ 
  void EnterWithScope();
  
  /*! \brief Exclude databse from current thread localcontext. */ 
  void ExitWithScope();
};

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
