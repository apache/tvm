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
 * \brief Runtime part of tuning capabilities for cublas matmul primitives. 
 */
#include <tvm/node/node.h>

#include <dmlc/json.h>
#include <cublasLt.h>

namespace tvm {
namespace contrib {

/*! \brief The descriptor of cublas Matmul Algo. */
class AlgoDescNode : public runtime::Object {
 public:
  /*! \brief Estimated execution time of kernel. */
  double estimated_time_us;
  /*! \brief Original cublas algo object. */
  cublasLtMatmulAlgo_t algo;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("estimated_time_us", &estimated_time_us);
    // `algo` is not visited
  }

  static constexpr const char* _type_key = "contrib.cublas.AlgoDesc";
  TVM_DECLARE_BASE_OBJECT_INFO(AlgoDescNode, runtime::Object);

 public:
  /*! \brief Default destructor. */
  virtual ~AlgoDescNode() = default;
  
  /*! \brief JSON serialization routine. */
  void Save(dmlc::JSONWriter* writer) const;
  
  /*! \brief JSON deserialization routine. */
  void Load(dmlc::JSONReader* reader);
};

/*!
 * \brief Managed reference to AlgoDescNode
 * \sa AlgoDescNode
 */
class AlgoDesc : public runtime::ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AlgoDesc, runtime::ObjectRef, AlgoDescNode);

  /*! \brief Default constructor. */
  AlgoDesc();
  
  /*! \brief Constructor with args. 
   *
   * \param estimated_time_us Measured or estimated execution time in micro seconds.
   * \param algo Original algo object.
   */
  AlgoDesc(const double estimated_time_us, const cublasLtMatmulAlgo_t &algo);
};

/*! \brief Colelction of AlgoDesc arranged by batch size. */
class AlgoCollectionNode : public runtime::Object {
 public:
  /*! \brief Algorithms available in this collection. */
  std::vector<AlgoDesc> algos;

  /*! \brief Mapping of algorithm to proper range of dynamic variable. 
   *   
   *  Elements meaning:
   *  [0] - end_of_range. Highest value of dinamic variable range which should 
   *        use this particular algorithm object.
   *  [1] - algo_idx. Index of algorithm in algos array.
   */  
  std::vector<std::pair<size_t, size_t>> regions;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `algos` is not visited
    // `regions` is not visited
  }

  static constexpr const char* _type_key = "contrib.cublas.AlgoCollection";
  TVM_DECLARE_BASE_OBJECT_INFO(AlgoCollectionNode, runtime::Object);

  /*! \brief Get proper algo for particular batch size */
  AlgoDesc GetAlgoFor(size_t dyn_dim_val);
  
  /*! \brief JSON serialization routine. */
  void Save(dmlc::JSONWriter* writer) const;
    
  /*! \brief JSON deserialization routine. */
  void Load(dmlc::JSONReader* reader);
};

/*!
 * \brief Managed reference to AlgoCollectionNode
 * \sa AlgoCollectionNode
 */
class AlgoCollection : public runtime::ObjectRef {
 public:
  /*!
   * \brief Construct algo collection from json
   * \param json The string with json representation.
   * \return Resulting algo collection.
   */  
  TVM_DLL static AlgoCollection FromJSON(const std::string& data);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AlgoCollection, runtime::ObjectRef, AlgoCollectionNode);

  /*! \brief Default constructor */
  AlgoCollection();
  
  /*! \brief Get operator wrapper */
  AlgoDesc operator() (int dyn_dim_val) const {
    return get()->GetAlgoFor(dyn_dim_val);
  }
};

}  // namespace contrib
}  // namespace tvm


template<>
struct dmlc::json::Handler<tvm::contrib::AlgoDesc> {
  inline static void Write(dmlc::JSONWriter *writer, const tvm::contrib::AlgoDesc &value) {
    value->Save(writer);
  }
  inline static void Read(dmlc::JSONReader *reader, tvm::contrib::AlgoDesc *value) {
    value->operator->()->Load(reader);
  }
};

template<>
struct dmlc::json::Handler<tvm::contrib::AlgoCollection> {
  inline static void Write(dmlc::JSONWriter *writer, const tvm::contrib::AlgoCollection &value) {
    value->Save(writer);
  }
  inline static void Read(dmlc::JSONReader *reader, tvm::contrib::AlgoCollection *value) {
    value->operator->()->Load(reader);
  }
};
