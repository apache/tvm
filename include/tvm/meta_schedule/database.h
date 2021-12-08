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
#ifndef TVM_META_SCHEDULE_DATABASE_H_
#define TVM_META_SCHEDULE_DATABASE_H_

#include <tvm/meta_schedule/arg_info.h>
#include <tvm/target/target.h>
#include <tvm/tir/schedule/trace.h>

namespace tvm {
namespace meta_schedule {

/*! \brief A workload, i.e. an IRModule and its structural hash. */
class WorkloadNode : public runtime::Object {
 public:
  /*! \brief The type of structural hash */
  using THashCode = size_t;
  /*! \brief The workload's IRModule. */
  IRModule mod;
  /*! \brief The workload's structural hash. */
  THashCode shash;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("mod", &mod);
    // `shash` is not visited because TVM FFI doesn't support uint64_t
  }

  static constexpr const char* _type_key = "meta_schedule.Workload";
  TVM_DECLARE_FINAL_OBJECT_INFO(WorkloadNode, runtime::Object);

  /*!
   * \brief Export the workload to a JSON string.
   * \return An array containing the structural hash and the base64 json string.
   */
  ObjectRef AsJSON() const;
};

/*!
 * \brief Managed reference to WorkloadNode.
 *  \sa WorkloadNode
 */
class Workload : public runtime::ObjectRef {
 public:
  using THashCode = WorkloadNode::THashCode;
  /*!
   * \brief Constructor of Workload.
   * \param mod The workload's IRModule.
   */
  TVM_DLL explicit Workload(IRModule mod);
  /*!
   * \brief Constructor of Workload.
   * \param mod The workload's IRModule.
   * \param shash The workload's structural hash.
   */
  TVM_DLL explicit Workload(IRModule mod, THashCode shash);
  /*!
   * \brief Create a workload from a json object.
   * \param json_obj The json object.
   * \return The created workload.
   */
  TVM_DLL static Workload FromJSON(const ObjectRef& json_obj);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Workload, runtime::ObjectRef, WorkloadNode);
};

/*! \brief The hash method for Workload */
struct WorkloadHash {
  size_t operator()(const Workload& a) const { return a->shash; }
};

/*! \brief The equality check for Workload */
struct WorkloadEqual {
  bool operator()(const Workload& a, const Workload& b) const {
    return a->shash == b->shash && tvm::StructuralEqual()(a->mod, b->mod);
  }
};

/*! \brief The class of tuning records. */
class TuningRecordNode : public runtime::Object {
 public:
  /*! \brief The trace tuned. */
  tir::Trace trace;
  /*! \brief The profiling result in seconds. */
  Array<FloatImm> run_secs;
  /*! \brief The workload. */
  Workload workload{nullptr};
  /*! \brief The target for tuning. */
  Target target;
  /*! \brief The argument information. */
  Array<ArgInfo> args_info;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("trace", &trace);
    v->Visit("run_secs", &run_secs);
    v->Visit("workload", &workload);
    v->Visit("target", &target);
    v->Visit("args_info", &args_info);
  }

  static constexpr const char* _type_key = "meta_schedule.TuningRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuningRecordNode, runtime::Object);

  /*!
   * \brief Export the tuning record to a JSON string.
   * \return An array containing the trace, running secs, serialized target, and
   * argument information.
   */
  ObjectRef AsJSON() const;
};

/*!
 * \brief The managed reference of TuningRecordNode.
 * \sa TuningRecordNode
 */
class TuningRecord : public runtime::ObjectRef {
 public:
  /*!
   \brief Constructor of a tuning record.
   \param trace The trace of the tuning record.
   \param run_secs The running time of the tuning record.
   \param workload The workload of the tuning record.
   \param target The target of the tuning record.
   \param args_info The argument information of the tuning record.
  */
  TVM_DLL explicit TuningRecord(tir::Trace trace, Array<FloatImm> run_secs, Workload workload,
                                Target target, Array<ArgInfo> args_info);
  /*!
   * \brief Create a tuning record from a json object.
   * \param json_obj The json object.
   * \param workload The workload.
   * \return The tuning record created.
   */
  TVM_DLL static TuningRecord FromJSON(const ObjectRef& json_obj, const Workload& workload);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TuningRecord, runtime::ObjectRef, TuningRecordNode);
};

/* \brief The abstract interface of database. */
class DatabaseNode : public runtime::Object {
 public:
  /*! \brief Default destructor */
  virtual ~DatabaseNode() = default;
  /*!
   * \brief Look up or add workload to the database if missing.
   * \param mod The IRModule to be searched for or added.
   * \return The workload corresponding to the given IRModule.
   */
  virtual Workload CommitWorkload(const IRModule& mod) = 0;
  /*!
   * \brief Add a tuning record to the database.
   * \param record The tuning record to be added.
   */
  virtual void CommitTuningRecord(const TuningRecord& record) = 0;
  /*!
   * \brief Get the top K tuning records of given workload from the database.
   * \param workload The workload to be searched for.
   * \param top_k The number of top records to be returned.
   * \return An array of top K tuning records for the given workload.
   */
  virtual Array<TuningRecord> GetTopK(const Workload& workload, int top_k) = 0;
  /*!
   * \brief Get the size of the database.
   * \return The size of the database.
   */
  virtual int64_t Size() = 0;

  static constexpr const char* _type_key = "meta_schedule.Database";
  TVM_DECLARE_BASE_OBJECT_INFO(DatabaseNode, runtime::Object);
};

/*! \brief The database with customized methods on the python-side. */
class PyDatabaseNode : public DatabaseNode {
 public:
  /*!
   * \brief The function type of `CommitWorkload` method.
   * \param mod The IRModule to be searched for or added.
   * \return The workload corresponding to the given IRModule.
   */
  using FCommitWorkload = runtime::TypedPackedFunc<Workload(const IRModule&)>;
  /*!
   * \brief The function type of `CommitTuningRecord` method.
   * \param record The tuning record to be added.
   */
  using FCommitTuningRecord = runtime::TypedPackedFunc<void(const TuningRecord&)>;
  /*!
   * \brief The function type of `GetTopK` method.
   * \param workload The workload to be searched for.
   * \param top_k The number of top records to be returned.
   * \return An array of top K tuning records for the given workload.
   */
  using FGetTopK = runtime::TypedPackedFunc<Array<TuningRecord>(const Workload&, int)>;
  /*!
   * \brief The function type of `Size` method.
   * \return The size of the database.
   */
  using FSize = runtime::TypedPackedFunc<int64_t()>;

  /*! \brief The packed function to the `CommitWorkload` function. */
  FCommitWorkload f_commit_workload;
  /*! \brief The packed function to the `CommitTuningRecord` function. */
  FCommitTuningRecord f_commit_tuning_record;
  /*! \brief The packed function to the `GetTopK` function. */
  FGetTopK f_get_top_k;
  /*! \brief The packed function to the `Size` function. */
  FSize f_size;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // PackedFuncs are all not visited, because the reflection system doesn't take care of them,
    // so it cannot be accessible on the python side. If there is such need from the future,
    // we can then add corresponding accessor methods to help access on python.
    //
    // `f_commit_workload` is not visited
    // `f_commit_tuning_record` is not visited
    // `f_get_top_k` is not visited
    // `f_size` is not visited
  }

  Workload CommitWorkload(const IRModule& mod) final {
    ICHECK(f_commit_workload != nullptr) << "PyDatabase's CommitWorkload method not implemented!";
    return f_commit_workload(mod);
  }

  void CommitTuningRecord(const TuningRecord& record) final {
    ICHECK(f_commit_tuning_record != nullptr)
        << "PyDatabase's CommitTuningRecord method not implemented!";
    f_commit_tuning_record(record);
  }

  Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    ICHECK(f_get_top_k != nullptr) << "PyDatabase's GetTopK method not implemented!";
    return f_get_top_k(workload, top_k);
  }

  int64_t Size() final {
    ICHECK(f_size != nullptr) << "PyDatabase's Size method not implemented!";
    return f_size();
  }

  static constexpr const char* _type_key = "meta_schedule.PyDatabase";
  TVM_DECLARE_FINAL_OBJECT_INFO(PyDatabaseNode, DatabaseNode);
};

/*!
 * \brief Managed reference to DatabaseNode.
 * \sa DatabaseNode
 */
class Database : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a default database that uses JSON file for tuning records.
   * \param path_workload The path to the workload table.
   * \param path_tuning_record The path to the database table.
   * \param allow_missing Whether to create new file when the given path is not found.
   */
  TVM_DLL static Database JSONDatabase(String path_workload, String path_tuning_record,
                                       bool allow_missing);
  /*!
   * \brief Create a database with customized methods on the python-side.
   * \param f_commit_workload The packed function of `CommitWorkload`.
   * \param f_commit_tuning_record The packed function of `CommitTuningRecord`.
   * \param f_get_top_k The packed function of `GetTopK`.
   * \param f_size The packed function of `Size`.
   * \return The created database.
   */
  TVM_DLL static Database PyDatabase(PyDatabaseNode::FCommitWorkload f_commit_workload,
                                     PyDatabaseNode::FCommitTuningRecord f_commit_tuning_record,
                                     PyDatabaseNode::FGetTopK f_get_top_k,
                                     PyDatabaseNode::FSize f_size);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Database, runtime::ObjectRef, DatabaseNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_DATABASE_H_
