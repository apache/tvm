/*!
 *  Copyright (c) 2017 by Contributors
 * \file thread_axis_args.h
 * \brief Extract thread axis configuration from TVMArgs.
 */
#ifndef TVM_RUNTIME_THREAD_AXIS_ARGS_H_
#define TVM_RUNTIME_THREAD_AXIS_ARGS_H_

#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief workload speccification */
struct ThreadWorkLoad {
  // array, first three are thread configuration.
  size_t work_size[6];
  /*!
   * \param i The block dimension.
   * \return i-th block dim
   */
  inline size_t block_dim(size_t i) const {
    return work_size[i];
  }
  /*!
   * \param i The grid dimension.
   * \return i-th grid dim
   */
  inline size_t grid_dim(size_t i) const {
    return work_size[i + 3];
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
      if (tag == "threadIdx.x") {
        arg_index_map_.push_back(0);
        filled[0] = true;
      } else if (tag == "threadIdx.y") {
        arg_index_map_.push_back(1);
        filled[1] = true;
      } else if (tag == "threadIdx.z") {
        arg_index_map_.push_back(2);
        filled[2] = true;
      } else if (tag == "blockIdx.x") {
        arg_index_map_.push_back(3 + 0);
        filled[3] = true;
      } else if (tag == "blockIdx.y") {
        arg_index_map_.push_back(3 + 1);
        filled[3 + 1] = true;
      } else if (tag == "blockIdx.z") {
        arg_index_map_.push_back(3 + 2);
        filled[3 + 2] = true;
      } else {
        LOG(FATAL) << "do not known thread_tag=" << tag;
      }
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
#endif  // TVM_RUNTIME_THREAD_AXIS_ARGS_H_
