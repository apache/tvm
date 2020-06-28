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
 * \file ansor/utils.h
 * \brief Common utilities.
 */

#ifndef TVM_ANSOR_UTILS_H_
#define TVM_ANSOR_UTILS_H_

#include <dmlc/common.h>
#include <tvm/tir/expr.h>
#include <unordered_map>
#include <thread>
#include <deque>
#include <numeric>
#include <exception>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>
#include <string>
#include <future>
#include <tuple>
#include <set>

namespace std {

/*! \brief Hash function for std::pair */
template <typename T1, typename T2>
struct hash<std::pair<T1, T2> > {
  std::size_t operator()(const std::pair<T1, T2>& k) const {
    return ::dmlc::HashCombine(std::hash<T1>()(k.first), std::hash<T2>()(k.second));
  }
};

/*! \brief Hash function for std::tuple */
template <typename T1, typename T2, typename T3>
struct hash<std::tuple<T1, T2, T3> > {
  std::size_t operator()(const std::tuple<T1, T2, T3>& k) const {
    return ::dmlc::HashCombine(
        ::dmlc::HashCombine(std::hash<T1>()(std::get<0>(k)), std::hash<T2>()(std::get<1>(k))),
        std::hash<T3>()(std::get<2>(k)));
  }
};

/*! \brief Hash function for std::vector */
template <typename T>
struct hash<std::vector<T> > {
  std::size_t operator()(const std::vector<T>& vec) const {
    if (vec.empty()) {
      return 0;
    }
    std::size_t ret = std::hash<T>()(vec[0]);
    for (size_t i = 1; i < vec.size(); ++i) {
      ret = ::dmlc::HashCombine(ret, std::hash<T>()(vec[i]));
    }
    return ret;
  }
};

}  // namespace std

namespace tvm {
namespace ansor {

/********** Utilities for std::vector, std::set, std::string **********/
/*! \brief Get the first appearance index of elements in a vector */
template <typename T>
inline void GetIndices(const std::vector<T>& array,
                       const std::vector<T>& to_locate,
                       std::vector<int>* indices) {
  for (const auto& v : to_locate) {
    auto it = std::find(array.begin(), array.end(), v);
    if (it != array.end()) {
      indices->push_back(it - array.begin());
    } else {
      LOG(FATAL) << "Cannot find the item";
    }
  }
}

/*! \brief Get the first appearance index of an element in a vector */
template <typename T>
inline int GetIndex(const std::vector<T>& array, const T& to_locate) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (array[i] == to_locate) {
      return i;
    }
  }
  LOG(FATAL) << "Cannot find the item";
  return -1;
}

/*! \brief Delete an element in a vector */
template <typename T>
inline void DeleteItem(std::vector<T>* array, const T& to_delete) {
  auto iter = std::find(array->begin(), array->end(), to_delete);
  if (iter != array->end()) {
    array->erase(iter);
  }
}

/*! \brief Replace a sub-string to another sub-string in a string */
inline void StrReplace(std::string* base, const std::string& from, const std::string& to) {
  auto pos = base->find(from);
  while (pos != std::string::npos) {
    base->replace(pos, from.size(), to);
    pos = base->find(from, pos + to.size());
  }
}

/********** Utilities for TVM Containers / ByteArray **********/
/*! \brief Compute mean of a FloatImm array */
inline double FloatArrayMean(const Array<PrimExpr>& float_array) {
  double sum = 0;
  if (float_array.empty()) {
      return 0.0;
  }

  for (const auto&x : float_array) {
    auto floatimm = x.as<tir::FloatImmNode>();
    CHECK(floatimm != nullptr);
    sum += floatimm->value;
  }
  return sum / float_array.size();
}

/********** Other Utilities **********/
/*! \brief  Get an int value from an Expr */
inline int64_t GetIntImm(const PrimExpr& expr) {
  auto pint = expr.as<IntImmNode>();
  CHECK(pint != nullptr);
  return pint->value;
}

/*! \brief  Compute the product of the lengths of axes */
inline int64_t AxisLengthProd(const Array<tir::IterVar>& axes) {
  int64_t ret = 1.0;
  for (const auto& x : axes) {
    if (const IntImmNode* imm = x->dom->extent.as<IntImmNode>()) {
      ret *= imm->value;
    } else {
      return -1.0;
    }
  }
  return ret;
}

/*!
 * \brief Clean the name of an iterator to make it valid in python code.
 * \param str The original name.
 * \return The cleaned name.
 */
inline std::string CleanName(const std::string& str) {
  std::string ret = str;
  StrReplace(&ret, ".", "_");
  StrReplace(&ret, "@", "_");
  StrReplace(&ret, "outer", "o");
  StrReplace(&ret, "inner", "i");
  return ret;
}

/*! \brief  An empty output stream */
class NullStream : public std::ostream {
 public:
  NullStream() : std::ostream(nullptr) {}
  NullStream(const NullStream &) : std::ostream(nullptr) {}
  static NullStream& Global();
};

template <class T>
NullStream& operator<<(NullStream& os, const T& value) {
  return os;
}

/*! \brief Get std cout with verbose control */
inline std::ostream& StdCout(int verbose) {
  if (verbose >= 1) {
    return std::cout;
  } else {
    return NullStream::Global();
  }
}

/*! \brief Print a title */
inline void PrintTitle(const std::string& title, int verbose) {
  if (verbose >= 1) {
    std::cout << "------------------------------------------------------------" << "\n";
    std::cout << "-----------------------  [ " << title << " ]\n";
    std::cout << "------------------------------------------------------------" << std::endl;
  }
}

/*! \brief A simple thread pool */
class ThreadPool {
 public:
  void Launch(size_t n = 1) {
    for (std::size_t i = 0; i < n; ++i) {
      threads_.emplace_back([this] {WorkerFunc();});
    }
  }

  void BeginBatch(int n) {
    finish_ct_ = n;
    is_finished_ = n <= 0;
  }

  template<typename F, typename... Args, typename R = typename std::result_of<F(Args...)>::type>
  std::future<R> Enqueue(F&& f, Args&&... args) {
    std::packaged_task<R()> p(std::bind(f, args...));

    auto r = p.get_future();
    {
      std::unique_lock<std::mutex> l(m_);
      work_.emplace_back(std::move(p));
    }
    work_signal_.notify_one();
    return r;
  }

  void WaitBatch() {
    std::unique_lock<std::mutex> l(finish_mutex_);
    if (!is_finished_) {
      finish_signal_.wait(l);
    }
  }

  void Abort() {
    CancelPending();
    Join();
  }

  void CancelPending() {
    std::unique_lock<std::mutex> l(m_);
    work_.clear();
  }

  void Join() {
    {
      std::unique_lock<std::mutex> l(m_);
      for (size_t i = 0; i < threads_.size(); ++i) {
        work_.push_back({});
      }
    }
    work_signal_.notify_all();
    for (auto& t : threads_) {
      t.join();
    }
    threads_.clear();
  }

  size_t NumWorkers() {
    return threads_.size();
  }

  static const int REFRESH_EVERY = 128;
  static ThreadPool& Global();

  ~ThreadPool() {
    Join();
  }

 private:
  void WorkerFunc() {
    while (true) {
      std::packaged_task<void()> f;
      {
        std::unique_lock<std::mutex> l(m_);
        if (work_.empty()) {
          work_signal_.wait(l, [&]{ return !work_.empty(); });
        }
        f = std::move(work_.front());
        work_.pop_front();
      }
      if (!f.valid()) { return; }
      f();

      finish_ct_--;
      if (finish_ct_ == 0) {
        std::unique_lock<std::mutex> l(finish_mutex_);

        is_finished_ = true;
        finish_signal_.notify_one();
      }
    }
  }

  std::mutex m_;
  std::condition_variable work_signal_;
  std::deque<std::packaged_task<void()>> work_;
  std::vector<std::thread> threads_;

  bool is_finished_;
  std::mutex finish_mutex_;
  std::atomic<int> finish_ct_;
  std::condition_variable finish_signal_;
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_UTILS_H_
