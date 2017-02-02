/*!
 *  Copyright (c) 2017 by Contributors
 * \file thread_storage_scope.h
 * \brief Extract thread axis configuration from TVMArgs.
 */
#ifndef TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
#define TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_

#include <tvm/runtime/packed_func.h>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief class to represent storage scope */
struct StorageScope {
  /*! \brief The rank of the storage */
  int rank{0};
  // comparator
  inline bool operator==(const StorageScope& other) const {
    return rank == other.rank;
  }
  inline std::string to_string() const {
    switch (rank) {
      case 0: return "global";
      case 1: return "shared";
      case 2: return "local";
      default: LOG(FATAL) << "unknown storage scope"; return "";
    }
  }
  /*!
   * \brief make storage scope from string
   * \param s The string to be parsed.
   * \return The storage scope.
   */
  static StorageScope make(const std::string& s) {
    StorageScope r;
    if (s == "global") {
      r.rank = 0;
    } else if (s == "shared") {
      r.rank = 1;
    } else if (s == "local") {
      r.rank = 2;
    } else {
      LOG(FATAL) << "unknown storage scope " << s;
    }
    return r;
  }
};

/*! \brief class to represent thread scope */
struct ThreadScope {
  /*! \brief The rank of thread scope */
  int rank{0};
  /*! \brief the dimension index under the rank */
  int dim_index{0};
  /*!
   * \brief make storage scope from string
   * \param s The string to be parsed.
   * \return The storage scope.
   */
  static ThreadScope make(const std::string& s) {
    ThreadScope r;
    if (s.compare(0, 9, "blockIdx.") == 0) {
      r.rank = 0;
      r.dim_index = static_cast<int>(s[9] - 'x');
    } else if (s.compare(0, 10, "threadIdx.") == 0) {
      r.rank = 1;
      r.dim_index = static_cast<int>(s[10] - 'x');
    } else {
      LOG(FATAL) << "Unknown threadscope " << s;
    }
    return r;
  }
};


/*! \brief workload speccification */
struct ThreadWorkLoad {
  // array, first three are thread configuration.
  size_t work_size[6];
  /*!
   * \param i The block dimension.
   * \return i-th block dim
   */
  inline size_t block_dim(size_t i) const {
    return work_size[i + 3];
  }
  /*!
   * \param i The grid dimension.
   * \return i-th grid dim
   */
  inline size_t grid_dim(size_t i) const {
    return work_size[i];
  }
};
/*! \brief Thread axis configuration */
class ThreadAxisConfig {
 public:
  void Init(size_t base,
            const std::vector<std::string>& thread_axis_tags)  {
    base_ = base;
    std::vector<bool> filled(6, false);
    for (size_t i = 0; i < thread_axis_tags.size(); ++i) {
      const std::string& tag = thread_axis_tags[i];
      ThreadScope ts = ThreadScope::make(tag);
      arg_index_map_.push_back(ts.rank * 3 + ts.dim_index);
      filled[ts.rank * 3 + ts.dim_index] = true;
    }
    work_dim_ = 3;
    for (int i = 0; i < 3; ++i) {
      if (!filled[i]) {
        for (int j = i; j < 3; ++j) {
          CHECK(!filled[j] && !filled[j + 3])
              << "Invalid thread group configuration";
        }
        work_dim_ = i;
        break;
      } else {
        CHECK(filled[i])
            << "Must have both threadIdx and blockIdx";
      }
    }
  }
  // extract workload from arguments.
  ThreadWorkLoad Extract(TVMArgs x) const {
    ThreadWorkLoad w;
    std::fill(w.work_size, w.work_size + 6, 1);
    for (size_t i = 0; i < arg_index_map_.size(); ++i) {
      w.work_size[arg_index_map_[i]] =
          static_cast<size_t>(x.values[base_ + i].v_int64);
    }
    return w;
  }
  // return the work dim
  size_t work_dim() const {
    return work_dim_;
  }

 private:
  /*! \brief base axis */
  size_t base_;
  /*! \brief The worker dimension */
  size_t work_dim_;
  /*! \brief The index mapping. */
  std::vector<uint32_t> arg_index_map_;
};

}  // namespace runtime
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::runtime::StorageScope> {
  std::size_t operator()(const ::tvm::runtime::StorageScope& k) const {
    return static_cast<size_t>(k.rank);
  }
};
}  // namespace std
#endif  // TVM_RUNTIME_THREAD_STORAGE_SCOPE_H_
