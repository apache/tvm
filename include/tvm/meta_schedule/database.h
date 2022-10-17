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

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/meta_schedule/arg_info.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/trace.h>

#include <memory>

namespace tvm {
namespace meta_schedule {

class ModuleEquality;

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
  explicit WorkloadEqual(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}

  bool operator()(const Workload& a, const Workload& b) const;

 private:
  /*! \brief The module equality testing and hashing method */
  const ModuleEquality& mod_eq_;
};

/*! \brief The class of measure candidates. */
class MeasureCandidate;

/*! \brief The class of tuning records. */
class TuningRecordNode : public runtime::Object {
 public:
  /*! \brief The trace tuned. */
  tir::Trace trace;
  /*! \brief The workload. */
  Workload workload{nullptr};
  /*! \brief The profiling result in seconds. */
  Optional<Array<FloatImm>> run_secs;
  /*! \brief The target for tuning. */
  Optional<Target> target;
  /*! \brief The argument information. */
  Optional<Array<ArgInfo>> args_info;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("trace", &trace);
    v->Visit("workload", &workload);
    v->Visit("run_secs", &run_secs);
    v->Visit("target", &target);
    v->Visit("args_info", &args_info);
  }

  static constexpr const char* _type_key = "meta_schedule.TuningRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuningRecordNode, runtime::Object);

  /*! \brief Construct the measure candidate given the initial IR module and trace
   * stored in the tuning record. */
  MeasureCandidate AsMeasureCandidate() const;
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
   \param workload The workload of the tuning record.
   \param run_secs The running time of the tuning record.
   \param target The target of the tuning record.
   \param args_info The argument information of the tuning record.
  */
  TVM_DLL explicit TuningRecord(tir::Trace trace, Workload workload,
                                Optional<Array<FloatImm>> run_secs, Optional<Target> target,
                                Optional<Array<ArgInfo>> args_info);
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
  /*!
   * \brief Constructor
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   *  It must be one of the followings:
   *    - "structural": Use StructuralEqual/Hash
   *    - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
   *                        equality testing and hashing.
   */
  explicit DatabaseNode(String mod_eq_name = "structural");

  /*! \brief Default destructor */
  virtual ~DatabaseNode();
  /*!
   * \brief Check if the database has the given workload.
   * \param mod The IRModule to be searched for.
   * \return Whether the database has the given workload.
   */
  virtual bool HasWorkload(const IRModule& mod) = 0;
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
   * \brief Get all tuning records from the database.
   * \return An Array of all the tuning records in the database.
   */
  virtual Array<TuningRecord> GetAllTuningRecords() = 0;
  /*!
   * \brief Get the size of the database.
   * \return The size of the database.
   */
  virtual int64_t Size() = 0;
  /*!
   * \brief Query the best record of the given workload from the database.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The best record of the given workload; NullOpt if not found.
   */
  virtual Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                                   const String& workload_name);
  /*!
   * \brief Query the best schedule of the given workload from the database.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The schedule in the best schedule of the given workload; NullOpt if not found.
   */
  virtual Optional<tir::Schedule> QuerySchedule(const IRModule& mod, const Target& target,
                                                const String& workload_name);
  /*!
   * \brief Query the best IRModule of the given workload from the database.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The IRModule in the best IRModule of the given workload; NullOpt if not found.
   */
  virtual Optional<IRModule> QueryIRModule(const IRModule& mod, const Target& target,
                                           const String& workload_name);

  /*! \brief Return a reference to the owned module equality method instance. */
  const ModuleEquality& GetModuleEquality() const {
    ICHECK(mod_eq_);
    return *mod_eq_;
  }

  static constexpr const char* _type_key = "meta_schedule.Database";
  TVM_DECLARE_BASE_OBJECT_INFO(DatabaseNode, runtime::Object);

 private:
  /*! \brief The module equality testing and hashing method */
  std::unique_ptr<ModuleEquality> mod_eq_;
};

/*! \brief The database with customized methods on the python-side. */
class PyDatabaseNode : public DatabaseNode {
 public:
  /*!
   * \brief Constructor
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   *  It must be one of the followings:
   *    - "structural": Use StructuralEqual/Hash
   *    - "ignore-ndarray": Same as "structural", but ignore ndarray raw data during
   *                        equality testing and hashing.
   */
  explicit PyDatabaseNode(String mod_eq_name = "structural");

  /*!
   * \brief The function type of `HasWorkload` method.
   * \param mod The IRModule to be searched for.
   * \return Whether the database has the given workload.
   */
  using FHasWorkload = runtime::TypedPackedFunc<bool(const IRModule&)>;
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
   * \brief The function type of `GetAllTuningRecords` method.
   * \return An Array of all the tuning records in the database.
   */
  using FGetAllTuningRecords = runtime::TypedPackedFunc<Array<TuningRecord>()>;
  /*!
   * \brief The function type of `QueryTuningRecord` method.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The best record of the given workload; NullOpt if not found.
   */
  using FQueryTuningRecord = runtime::TypedPackedFunc<Optional<TuningRecord>(
      const IRModule&, const Target&, const String&)>;
  /*!
   * \brief The function type of `QuerySchedule` method.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The schedule in the best schedule of the given workload; NullOpt if not found.
   */
  using FQuerySchedule = runtime::TypedPackedFunc<Optional<tir::Schedule>(
      const IRModule&, const Target&, const String&)>;
  /*!
   * \brief The function type of `QueryIRModule` method.
   * \param mod The IRModule to be searched for.
   * \param target The target to be searched for.
   * \param workload_name The name of the workload to be searched for.
   * \return The IRModule in the best IRModule of the given workload; NullOpt if not found.
   */
  using FQueryIRModule =
      runtime::TypedPackedFunc<Optional<IRModule>(const IRModule&, const Target&, const String&)>;
  /*!
   * \brief The function type of `Size` method.
   * \return The size of the database.
   */
  using FSize = runtime::TypedPackedFunc<int64_t()>;

  /*! \brief The packed function to the `HasWorkload` function. */
  FHasWorkload f_has_workload;
  /*! \brief The packed function to the `CommitWorkload` function. */
  FCommitWorkload f_commit_workload;
  /*! \brief The packed function to the `CommitTuningRecord` function. */
  FCommitTuningRecord f_commit_tuning_record;
  /*! \brief The packed function to the `GetTopK` function. */
  FGetTopK f_get_top_k;
  /*! \brief The packed function to the `GetAllTuningRecords` function. */
  FGetAllTuningRecords f_get_all_tuning_records;
  /*! \brief The packed function to the `QueryTuningRecord` function. */
  FQueryTuningRecord f_query_tuning_record;
  /*! \brief The packed function to the `QuerySchedule` function. */
  FQuerySchedule f_query_schedule;
  /*! \brief The packed function to the `QueryIRModule` function. */
  FQueryIRModule f_query_ir_module;
  /*! \brief The packed function to the `Size` function. */
  FSize f_size;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // PackedFuncs are all not visited, because the reflection system doesn't take care of them,
    // so it cannot be accessible on the python side. If there is such need from the future,
    // we can then add corresponding accessor methods to help access on python.
    // `f_has_workload` is not visited
    // `f_commit_workload` is not visited
    // `f_commit_tuning_record` is not visited
    // `f_get_top_k` is not visited
    // `f_get_all_tuning_records` is not visited
    // `f_query_tuning_record` is not visited
    // `f_query_schedule` is not visited
    // `f_query_ir_module` is not visited
    // `f_size` is not visited
  }

  bool HasWorkload(const IRModule& mod) final {
    ICHECK(f_has_workload != nullptr) << "PyDatabase's HasWorkload method not implemented!";
    return f_has_workload(mod);
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

  Array<TuningRecord> GetAllTuningRecords() final {
    ICHECK(f_get_all_tuning_records != nullptr)
        << "PyDatabase's GetAllTuningRecords method not implemented!";
    return f_get_all_tuning_records();
  }

  Optional<TuningRecord> QueryTuningRecord(const IRModule& mod, const Target& target,
                                           const String& workload_name) final {
    if (f_query_tuning_record == nullptr) {
      return DatabaseNode::QueryTuningRecord(mod, target, workload_name);
    } else {
      return f_query_tuning_record(mod, target, workload_name);
    }
  }

  Optional<tir::Schedule> QuerySchedule(const IRModule& mod, const Target& target,
                                        const String& workload_name) final {
    if (f_query_schedule == nullptr) {
      return DatabaseNode::QuerySchedule(mod, target, workload_name);
    } else {
      return f_query_schedule(mod, target, workload_name);
    }
  }

  Optional<IRModule> QueryIRModule(const IRModule& mod, const Target& target,
                                   const String& workload_name) final {
    if (f_query_ir_module == nullptr) {
      return DatabaseNode::QueryIRModule(mod, target, workload_name);
    } else {
      return f_query_ir_module(mod, target, workload_name);
    }
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
   * \brief An in-memory database.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   */
  TVM_DLL static Database MemoryDatabase(String mod_eq_name = "structural");
  /*!
   * \brief A database for injecting handcrafted schedule functions.
   * \param schedule_fn The function to do scheduling, which takes a TIR schedule,
   * and returns a boolean indicating if the schedule is successful.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   */
  TVM_DLL static Database ScheduleFnDatabase(
      runtime::TypedPackedFunc<bool(tir::Schedule)> schedule_fn, String mod_eq_name = "structural");
  /*!
   * \brief Create a default database that uses JSON file for tuning records.
   * \param path_workload The path to the workload table.
   * \param path_tuning_record The path to the database table.
   * \param allow_missing Whether to create new file when the given path is not found.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   */
  TVM_DLL static Database JSONDatabase(String path_workload, String path_tuning_record,
                                       bool allow_missing, String mod_eq_name = "structural");
  /*!
   * \brief A database composed of multiple databases, allowing users to guide IR rewriting using
   * combined knowledge of those databases. To each query, it returns the best record among all the
   * databases given.
   * \param databases The list of databases to be combined.
   * \return The combined database.
   */
  TVM_DLL static Database UnionDatabase(Array<Database, void> databases);
  /*!
   * \brief A database composed of multiple databases, allowing users to guide IR rewriting using
   * combined knowledge of those databases. To each query, it returns the record from the first
   * database that responds to the query.
   * \param databases The database to be subsetted.
   * \return The subsetted database.
   */
  TVM_DLL static Database OrderedUnionDatabase(Array<Database, void> databases);
  /*!
   * \brief Create a database with customized methods on the python-side.
   * \param f_has_workload The packed function of `HasWorkload`.
   * \param f_commit_workload The packed function of `CommitWorkload`.
   * \param f_commit_tuning_record The packed function of `CommitTuningRecord`.
   * \param f_get_top_k The packed function of `GetTopK`.
   * \param f_get_all_tuning_records The packed function of `GetAllTuningRecords`.
   * \param f_query_tuning_record The packed function of `QueryTuningRecord`.
   * \param f_query_schedule The packed function of `QuerySchedule`.
   * \param f_query_ir_module The packed function of `QueryIRModule`.
   * \param f_size The packed function of `Size`.
   * \param mod_eq_name A string to specify the module equality testing and hashing method.
   * \return The created database.
   */
  TVM_DLL static Database PyDatabase(PyDatabaseNode::FHasWorkload f_has_workload,
                                     PyDatabaseNode::FCommitWorkload f_commit_workload,
                                     PyDatabaseNode::FCommitTuningRecord f_commit_tuning_record,
                                     PyDatabaseNode::FGetTopK f_get_top_k,
                                     PyDatabaseNode::FGetAllTuningRecords f_get_all_tuning_records,
                                     PyDatabaseNode::FQueryTuningRecord f_query_tuning_record,
                                     PyDatabaseNode::FQuerySchedule f_query_schedule,
                                     PyDatabaseNode::FQueryIRModule f_query_ir_module,
                                     PyDatabaseNode::FSize f_size,
                                     String mod_eq_name = "structural");
  /*! \return The current Database in the scope. */
  static Optional<Database> Current();
  /*! \brief Entering the scope of the context manager */
  void EnterWithScope();
  /*! \brief Exiting the scope of the context manager */
  void ExitWithScope();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Database, runtime::ObjectRef, DatabaseNode);
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_DATABASE_H_
