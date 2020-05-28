/*!
 *  Copyright (c) 2020 by Contributors
 * \file ansor/utils.h
 * \brief Common utilities
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

// hash function for std::pair, std::vector and std::tuple
template <typename T1, typename T2>
struct hash<std::pair<T1, T2> > {
  std::size_t operator()(const std::pair<T1, T2>& k) const {
    return ::dmlc::HashCombine(std::hash<T1>()(k.first), std::hash<T2>()(k.second));
  }
};

template <typename T1, typename T2, typename T3>
struct hash<std::tuple<T1, T2, T3> > {
  std::size_t operator()(const std::tuple<T1, T2, T3>& k) const {
    return ::dmlc::HashCombine(
        ::dmlc::HashCombine(std::hash<T1>()(std::get<0>(k)), std::hash<T2>()(std::get<1>(k))),
        std::hash<T3>()(std::get<2>(k)));
  }
};

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

/*! \brief Macro to make it easy to define node ref type given node */
#define TVM_DEFINE_NODE_REF(TypeName, NodeName)                     \
  class TypeName : public ObjectRef {                               \
   public:                                                          \
    TVM_DEFINE_OBJECT_REF_METHODS(TypeName, ObjectRef, NodeName);   \
  };                                                                \

/*! \brief Macro to make it easy to define mutable node ref type given node */
#define TVM_DEFINE_MUTABLE_NODE_REF(TypeName, NodeName)                      \
  class TypeName : public ObjectRef {                                        \
   public:                                                                   \
    TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeName, ObjectRef, NodeName);    \
  };                                                                         \

/*!
 * \brief Macro to make it easy to define node ref type that
 *  has a CopyOnWrite member function.
 */
#define TVM_DEFINE_COW_NODE_REF(TypeName, BaseType, NodeName)           \
  class TypeName : public BaseType {                                    \
   public:                                                              \
    TVM_DEFINE_OBJECT_REF_METHODS(TypeName, BaseType, NodeName);        \
    TVM_DEFINE_OBJECT_REF_COW_METHOD(NodeName);                         \
  };

/********** Utilities for std::vector, std::set **********/

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

/*! \brief Compute the product of all elements in a vector */
inline int64_t ElementProduct(const std::vector<int>& array) {
  int64_t ret = 1;
  for (auto x : array) {
    ret *= x;
  }
  return ret;
}

/* \brief Get the maximum element in a vector */
template <typename T>
T MaximumElement(const std::vector<T>& array) {
  CHECK(!array.empty());
  const T* pmax = &array[0];
  for (size_t i = 1; i < array.size(); ++i) {
    if (array[i] > *pmax) {
      pmax = &array[i];
    }
  }
  return *pmax;
}

/*! \brief Move elements from multiple vectors to one vector */
template<typename T>
std::vector<T>& ConcatenateMove(std::vector<T>* out, std::vector<T>* in) {
  out->insert(out->end(), std::make_move_iterator(in->begin()),
              std::make_move_iterator(in->end()));
  return *out;
}

/*! \brief Move elements from multiple vectors to one vector */
template<typename T, typename... Args>
std::vector<T>& ConcatenateMove(std::vector<T>* out, std::vector<T>* first, Args... args) {
  ConcatenateMove(out, first);
  ConcatenateMove(out, args...);
  return *out;
}

/* \brief Get a random permutation of integers [0, n-1] */
template <typename G>
void RandomPermutation(int n, std::vector<int>* out, G* gen) {
  out->assign(n, 0);
  std::iota(out->begin(), out->end(), 0);
  std::shuffle(out->begin(), out->end(), *gen);
}

/* \brief Random sample without replacement */
template <typename T, typename G>
void RandomSample(std::vector<T>* in_data, size_t out_size, G* gen) {
  // Note: This function is inefficient in the cases when out_size << in_data.size()
  out_size = std::min(in_data->size(), out_size);

  if (in_data->size() <= out_size) {  // return all
    return;
  }
  std::vector<int> indices;
  RandomPermutation(in_data->size(), &indices, gen);

  std::vector<T> tmp_data;
  tmp_data.reserve(out_size);
  for (size_t i = 0; i < out_size; ++i) {
    tmp_data.push_back(std::move((*in_data)[indices[i]]));
  }

  *in_data = std::move(tmp_data);
}

/*! \brief Argsort. Order: largest to smallest */
template <typename T>
inline void Argsort(const std::vector<T>& scores, std::vector<int>* index) {
  index->clear(); index->reserve(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    index->push_back(i);
  }
  auto cmp = [&scores](int l, int r) {
    return scores[l] > scores[r];
  };
  std::sort(index->begin(), index->end(), cmp);
}

// Do x++ for all x in the set such that x >= threshold
inline void SetAddOne(std::set<int>* set, int threshold = 0) {
  std::set<int> new_set;
  for (int x : *set) {
    if (x >= threshold) {
      new_set.insert(x + 1);
    } else {
      new_set.insert(x);
    }
  }
  *set = std::move(new_set);
}

// Compute Jaccard Similarity of two sets
template <typename T>
double JaccardSimilarity(std::set<T> s1, std::set<T> s2) {
  std::vector<T> intersect;
  std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                        std::back_inserter(intersect));
  return 1.0 * intersect.size() / (s1.size() + s2.size() - intersect.size());
}

/********** Utilities for std::string **********/

/*! Return whether a string ends with a another substring */
inline bool StrEndsWith(const std::string& a, const std::string& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
}

/*! Return whether a string starts with a another substring */
inline bool StrStartsWith(const std::string& a, const std::string& b) {
  if (b.size() > a.size()) return false;
  return std::equal(a.begin(), a.begin() + b.size(), b.begin());
}

/*! Replace a sub-string to another sub-string in a string */
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

/*! \brief Serialize a 2-dimensional vector to TVMByteArray.
 * This is used for sending data to python code */
template <typename T>
inline TVMByteArray Serialize2dVector(std::vector<std::vector<T> >&& in_data,
                                      std::vector<char>* out_data) {
  size_t total_bytes = 0;
  std::vector<int> size_vector;

  // serialize sizes
  total_bytes += (1 + in_data.size()) * sizeof(int);
  size_vector.reserve(in_data.size() + 1);
  size_vector.push_back(in_data.size());
  for (const auto& x : in_data) {
    size_vector.push_back(static_cast<int>(x.size()));
    total_bytes += sizeof(T) * x.size();
  }

  out_data->reserve(total_bytes);
  char* ptr = out_data->data();
  memmove(ptr, reinterpret_cast<char*>(size_vector.data()), (1 + in_data.size()) * sizeof(int));
  ptr += (1 + in_data.size()) * sizeof(int);

  // serialize in_data
  for (auto& x : in_data) {
    memmove(ptr, x.data(), sizeof(T) * x.size());
    ptr += sizeof(T) * x.size();
    x.clear();
  }

  CHECK_EQ(ptr - out_data->data(), total_bytes);

  return TVMByteArray{out_data->data(), total_bytes};
}

/********** Other Utilities **********/

// Get an int value from an Expr
inline int64_t GetIntImm(const PrimExpr& expr) {
  auto pint = expr.as<IntImmNode>();
  CHECK(pint != nullptr);
  return pint->value;
}


// Compute the product of the lengths of axes
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


// An empty output stream
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

/*!
 * \brief Enumerate all possible factorization schemes for splitting an axes.
 * \note This class will memorize the results for reuse.
 */
class SplitFactorizationMemo {
 public:
  using QueryKey = std::tuple<int, int, int>;

  const std::vector<std::vector<PrimExpr> >& GetFactorizationSchemes(
      int extent, int n_lengths, int max_innermost_factor);
  const std::vector<int>& GetFactors(int n);

 private:
  void DfsEnumerate(int now, int remaining_lenght, int max_innermost_factor);

  std::unordered_map<QueryKey, std::vector<std::vector<PrimExpr> > > memory_;

  int n_lengths_;
  std::vector<PrimExpr> tmp_stack_;
  std::vector<std::vector<PrimExpr> >* results_;
  std::unordered_map<int, std::vector<int>> factor_memory_;
};

}  // namespace ansor
}  // namespace tvm

#endif  // TVM_ANSOR_UTILS_H_
