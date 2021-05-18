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
 * \file src/ir/instrument.cc
 * \brief Infrastructure for instrumentation.
 */
#include <dmlc/thread_local.h>
#include <tvm/ir/instrument.h>
#include <tvm/ir/transform.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/registry.h>

#include <stack>

namespace tvm {
namespace instrument {

/*!
 * \brief A named PassInstrument implementation
 * \sa NamedPassInstrument
 */
class NamedPassInstrumentNode : public PassInstrumentNode {
 public:
  /*! \brief Name of this pass instrument object. */
  String name;

  /*! \brief Callback for instrumentation environment set up. */
  runtime::TypedPackedFunc<void()> set_up_callback;
  /*! \brief Callback for instrumentation environment clean up. */
  runtime::TypedPackedFunc<void()> tear_down_callback;

  /*! \brief Callback to run before a pass. */
  runtime::TypedPackedFunc<bool(const IRModule&, const transform::PassInfo&)>
      run_before_pass_callback;
  /*! \brief Callback to run after a pass. */
  runtime::TypedPackedFunc<void(const IRModule&, const transform::PassInfo&)>
      run_after_pass_callback;

  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  /*! \brief Set up environment for instrumentation. */
  void SetUp() const final;

  /*! \brief Clean up instrumentation environment. */
  void TearDown() const final;

  /*!
   * \brief Instrument before pass run, determine whether to run the pass or not.
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   *
   * \return true to run the pass; false to skip the pass.
   */
  bool RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const final;

  /*!
   * \brief Instrument after pass run.
   *
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   */
  void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const final;

  static constexpr const char* _type_key = "instrument.NamedPassInstrument";
  TVM_DECLARE_FINAL_OBJECT_INFO(NamedPassInstrumentNode, PassInstrumentNode);
};

/*!
 * \brief Managed reference class for NamedPassInstrumentNode
 * \sa NamedPassInstrumentNode
 */
class NamedPassInstrument : public PassInstrument {
 public:
  /*!
   * \brief Constructor
   * \param name Name for this instrumentation.
   */
  TVM_DLL NamedPassInstrument(String name);

  /*!
   * \brief mutable accessor.
   * \return mutable access pointer.
   */
  NamedPassInstrumentNode* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<NamedPassInstrumentNode*>(get_mutable());
  }

  TVM_DEFINE_OBJECT_REF_METHODS(NamedPassInstrument, PassInstrument, NamedPassInstrumentNode);
};

NamedPassInstrument::NamedPassInstrument(String name) {
  auto pi = make_object<NamedPassInstrumentNode>();
  pi->name = std::move(name);
  data_ = std::move(pi);
}

void NamedPassInstrumentNode::SetUp() const {
  if (set_up_callback != nullptr) {
    set_up_callback();
  }
}

void NamedPassInstrumentNode::TearDown() const {
  if (tear_down_callback != nullptr) {
    tear_down_callback();
  }
}

bool NamedPassInstrumentNode::RunBeforePass(const IRModule& ir_module,
                                            const transform::PassInfo& pass_info) const {
  if (run_before_pass_callback == nullptr) {
    return true;
  }

  return run_before_pass_callback(ir_module, pass_info);
}

void NamedPassInstrumentNode::RunAfterPass(const IRModule& ir_module,
                                           const transform::PassInfo& pass_info) const {
  if (run_after_pass_callback != nullptr) {
    run_after_pass_callback(ir_module, pass_info);
  }
}

TVM_REGISTER_NODE_TYPE(NamedPassInstrumentNode);

TVM_REGISTER_GLOBAL("instrument.NamedPassInstrument")
    .set_body_typed([](String name,
                       runtime::TypedPackedFunc<bool(const IRModule&, const transform::PassInfo&)>
                           run_before_pass,
                       runtime::TypedPackedFunc<void(const IRModule&, const transform::PassInfo&)>
                           run_after_pass,
                       runtime::TypedPackedFunc<void()> set_up,
                       runtime::TypedPackedFunc<void()> tear_down) {
      auto pi = NamedPassInstrument(name);
      pi->run_before_pass_callback = std::move(run_before_pass);
      pi->run_after_pass_callback = std::move(run_after_pass);

      pi->set_up_callback = std::move(set_up);
      pi->tear_down_callback = std::move(tear_down);
      return pi;
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<NamedPassInstrumentNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const NamedPassInstrumentNode*>(ref.get());
      p->stream << node->name;
    });

/*! \brief PassProfile stores profiling information for a given pass and its sub-passes. */
struct PassProfile {
  // TODO(@altanh): expose PassProfile through TVM Object API
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double, std::micro>;
  using Time = std::chrono::time_point<Clock>;

  /*! \brief The name of the pass being profiled. */
  String name;
  /*! \brief The time when the pass was entered. */
  Time start;
  /*! \brief The time when the pass completed. */
  Time end;
  /*! \brief The total duration of the pass, i.e. end - start. */
  Duration duration;
  /*! \brief PassProfiles for all sub-passes invoked during the execution of the pass. */
  std::vector<PassProfile> children;

  explicit PassProfile(String name)
      : name(name), start(Clock::now()), end(Clock::now()), children() {}

  /*! \brief Gets the PassProfile of the currently executing pass. */
  static PassProfile* Current();
  /*! \brief Pushes a new PassProfile with the given pass name. */
  static void EnterPass(String name);
  /*! \brief Pops the current PassProfile. */
  static void ExitPass();
};

struct PassProfileThreadLocalEntry {
  /*! \brief The placeholder top-level PassProfile. */
  PassProfile root;
  /*! \brief The stack of PassProfiles for nested passes currently running. */
  std::stack<PassProfile*> profile_stack;

  PassProfileThreadLocalEntry() : root("root") {}
};

/*! \brief Thread local store to hold the pass profiling data. */
typedef dmlc::ThreadLocalStore<PassProfileThreadLocalEntry> PassProfileThreadLocalStore;

void PassProfile::EnterPass(String name) {
  PassProfile* cur = PassProfile::Current();
  cur->children.emplace_back(name);
  PassProfileThreadLocalStore::Get()->profile_stack.push(&cur->children.back());
}

void PassProfile::ExitPass() {
  PassProfile* cur = PassProfile::Current();
  ICHECK_NE(cur->name, "root") << "mismatched enter/exit for pass profiling";
  cur->end = PassProfile::Clock::now();
  cur->duration = std::chrono::duration_cast<PassProfile::Duration>(cur->end - cur->start);
  PassProfileThreadLocalStore::Get()->profile_stack.pop();
}

PassProfile* PassProfile::Current() {
  PassProfileThreadLocalEntry* entry = PassProfileThreadLocalStore::Get();
  if (!entry->profile_stack.empty()) {
    return entry->profile_stack.top();
  } else {
    return &entry->root;
  }
}

String RenderPassProfiles() {
  PassProfileThreadLocalEntry* entry = PassProfileThreadLocalStore::Get();
  CHECK(entry->profile_stack.empty()) << "cannot print pass profile while still in a pass!";

  if (entry->root.children.empty()) {
    LOG(WARNING) << "no passes have been profiled, did you enable pass profiling?";
    return String();
  }

  // (depth, parent_duration, pass)
  std::stack<std::tuple<size_t, PassProfile::Duration, PassProfile*>> profiles;

  // push top level passes
  PassProfile::Duration top_dur(0);
  for (auto it = entry->root.children.begin(); it != entry->root.children.end(); ++it) {
    top_dur += it->duration;
  }
  for (auto it = entry->root.children.rbegin(); it != entry->root.children.rend(); ++it) {
    profiles.push(std::make_tuple(0, top_dur, &*it));
  }

  std::ostringstream os;
  os << std::fixed;

  while (profiles.size() > 0) {
    size_t depth;
    PassProfile::Duration parent_duration;
    PassProfile* profile;
    std::tie(depth, parent_duration, profile) = profiles.top();
    profiles.pop();

    // indent depth
    for (size_t i = 0; i < depth; ++i) {
      os << "\t";
    }

    // calculate time spent in pass itself (excluding sub-passes), and push children
    PassProfile::Duration self_duration = profile->duration;
    for (auto it = profile->children.rbegin(); it != profile->children.rend(); ++it) {
      self_duration -= it->duration;
      profiles.push(std::make_tuple(depth + 1, profile->duration, &*it));
    }

    double parent_pct = profile->duration.count() / parent_duration.count() * 100.0;
    double total_pct = profile->duration.count() / top_dur.count() * 100.0;

    os << profile->name << ": ";
    os << std::setprecision(0);
    os << profile->duration.count() << "us [" << self_duration.count() << "us] ";
    os << std::setprecision(2) << "(" << total_pct << "%; " << parent_pct << "%)\n";
  }

  return os.str();
}

TVM_REGISTER_GLOBAL("instrument.RenderTimePassProfiles").set_body_typed(RenderPassProfiles);

TVM_REGISTER_GLOBAL("instrument.MakePassesTimeInstrument").set_body_typed([]() {
  auto run_before_pass = [](const IRModule&, const transform::PassInfo& pass_info) {
    PassProfile::EnterPass(pass_info->name);
    return true;
  };

  auto run_after_pass = [](const IRModule&, const transform::PassInfo& pass_info) {
    PassProfile::ExitPass();
  };

  auto tear_down = []() { PassProfileThreadLocalStore::Get()->root.children.clear(); };

  auto pi = NamedPassInstrument("PassesTimeInstrument");
  pi->run_before_pass_callback = run_before_pass;
  pi->run_after_pass_callback = run_after_pass;

  pi->tear_down_callback = tear_down;
  return pi;
});

}  // namespace instrument
}  // namespace tvm
