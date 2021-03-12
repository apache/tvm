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

/**
 * \file
 * \brief C++ wrappers and helpers to handle BNNS objects
 */

#ifndef TVM_RUNTIME_CONTRIB_BNNS_BNNS_WRP_H_
#define TVM_RUNTIME_CONTRIB_BNNS_BNNS_WRP_H_

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {
namespace BNNS {

using Dim = size_t;
using Shape = std::vector<Dim>;
using Dtype = BNNSDataType;
using HDL = void*;

void* default_alloc(size_t size) { return malloc(size); }

void default_free(void* ptr) { free(ptr); }

/**
 * Main abstraction for tensor representation
 *
 * Contains buffer handler and common attributes like shape and dtype.
 */
class Tensor {
 public:
  Tensor() = delete;
  Tensor(Tensor&) = delete;

  Tensor(Shape shape, Dtype dtype, void* hdl) {
    auto rank = shape.size();
    ICHECK(rank < BNNS_MAX_TENSOR_DIMENSION);

    desc_ = {BNNSNDArrayFlags(0),
             getPlainLayout(rank),
             {},       // shape
             {},       // strides
             hdl,      // data handler
             dtype,    // data type
             nullptr,  // table_data (clustering case), is not used
             dtype,
             1.f,
             0.f};
    std::copy(shape.rbegin(), shape.rend(), std::begin(desc_.size));

    desc_.data = hdl;
    is_external_data = true;
  }

  ~Tensor() {
    if (desc_.data && !is_external_data) {
      default_free(desc_.data);
      desc_.data = nullptr;
    }
  }

  void allocate_memory() {
    if (desc_.data && !is_external_data) {
      default_free(desc_.data);
    }
    const size_t buff_size = getSize(desc_) * getElementSize(desc_);
    desc_.data = default_alloc(buff_size);
    ICHECK(desc_.data);
    is_external_data = false;
  }

  void* get_data_hdl() const { return desc_.data; }

  void set_data_hdl(void* hdl) {
    if (desc_.data && !is_external_data) {
      default_free(desc_.data);
      desc_.data = nullptr;
    }

    desc_.data = hdl;
    is_external_data = true;
  }

  const BNNSNDArrayDescriptor& get_desc() const { return desc_; }

  static BNNSDataLayout getPlainLayout(size_t rank) {
    ICHECK(rank <= BNNS_MAX_TENSOR_DIMENSION);
    return static_cast<BNNSDataLayout>((rank << 16) | 0x8001);
  }

  static size_t getRank(BNNSDataLayout layout) { return (layout & 0xF0000) >> 16; }

  static size_t getRank(BNNSNDArrayDescriptor desc) { return getRank(desc.layout); }

  static size_t getSize(BNNSNDArrayDescriptor desc) {
    auto rank = getRank(desc);
    return std::accumulate(desc.size, desc.size + rank, 1, std::multiplies<int>());
  }

  /** return size of element in bytes */
  static size_t getElementSize(Dtype dtype) { return (dtype & 0xFFFF) / 8; }

  /** return size of element in bytes */
  static size_t getElementSize(const BNNSNDArrayDescriptor& desc) {
    return getElementSize(desc.data_type);
  }

 private:
  bool is_external_data = false;
  BNNSNDArrayDescriptor desc_;
};

using TensorPtr = std::shared_ptr<Tensor>;

/**
 * Tensor View object which represent how provided BNNS::Tensor will be considered
 *
 * The single BNNS::Tensor can be treated in different form depend on particular primitive
 * expectation. More other some primitive supports only external form of batching. So we have
 * some abstraction to describe how primitive will handle provided tensor.
 *
 * Batched View
 *   View with extracted dimension as external batch value
 *   example: Tensor [2, 3, 224, 224] -> View [3, 224, 224] with ext batch 2
 *
 * Party View
 *   The collection of view on the same tensor, can be the same view or with some stride
 *   example: Tensor [6, 5, 3, 3] -> 3 x View [2, 5, 3, 3] with stride 45
 */
class TView {
 public:
  /** Make view on provided tensor as is */
  static TView as_is(const TensorPtr& origin) {
    TView res;
    res.origin_ = origin;
    res.view_desc_ = origin->get_desc();
    return res;
  }

  /** Extract outer dimension to separate batch field. TView will became batched view */
  TView extract_outer_dim() const {
    auto rank = Tensor::getRank(view_desc_);
    TView res = *this;
    res.batch_size_ = view_desc_.size[rank - 1];
    res.batch_stride_ =
        std::accumulate(view_desc_.size, view_desc_.size + rank - 1, 1, std::multiplies<>());
    res.view_desc_.size[rank - 1] = 0;
    res.view_desc_.layout = Tensor::getPlainLayout(rank - 1);
    return res;
  }

  /** Squeeze all dims equal 1 */
  TView squeeze(size_t min_rank = 1) const {
    auto rank = Tensor::getRank(view_desc_);
    size_t squeezed_shape[BNNS_MAX_TENSOR_DIMENSION] = {};
    size_t squeezed_rank = 0;
    for (int i = 0; i < rank; i++)
      if (view_desc_.size[i] != 1) squeezed_shape[squeezed_rank++] = view_desc_.size[i];

    if (min_rank > squeezed_rank) {
      std::fill(squeezed_shape + squeezed_rank, squeezed_shape + min_rank, 1);
      squeezed_rank = min_rank;
    }

    TView res = *this;
    std::copy(squeezed_shape, squeezed_shape + squeezed_rank, res.view_desc_.size);
    std::fill(res.view_desc_.size + squeezed_rank, res.view_desc_.size + rank, 0);
    res.view_desc_.layout = Tensor::getPlainLayout(squeezed_rank);
    return res;
  }

  /** Expand the shape of an array */
  TView expand_dims(std::vector<size_t> axes) const {
    auto rank = Tensor::getRank(view_desc_);
    TView res = *this;
    size_t unsqueezed_shape[BNNS_MAX_TENSOR_DIMENSION] = {};
    size_t unsqueezed_rank = axes.size() + rank;
    ICHECK_LE(unsqueezed_rank, BNNS_MAX_TENSOR_DIMENSION);
    for (const auto& axis : axes) {
      ICHECK_LT(axis, unsqueezed_rank);
      unsqueezed_shape[axis] = 1;
    }
    for (int i = 0, orig_idx = 0; i < unsqueezed_rank; ++i) {
      if (unsqueezed_shape[i] == 1) continue;
      unsqueezed_shape[i] = view_desc_.size[orig_idx++];
    }
    std::copy(unsqueezed_shape, unsqueezed_shape + unsqueezed_rank, res.view_desc_.size);
    res.view_desc_.layout = Tensor::getPlainLayout(unsqueezed_rank);
    return res;
  }

  /** Unsqueeze tensor to a new rank */
  TView unsqueeze(size_t new_rank) const {
    ICHECK_LE(new_rank, BNNS_MAX_TENSOR_DIMENSION);
    auto rank = Tensor::getRank(view_desc_);
    ICHECK_GT(new_rank, rank);
    std::vector<size_t> axes(new_rank - rank);
    std::iota(axes.begin(), axes.end(), rank);
    return expand_dims(axes);
  }

  /** Construct new TView with specified layout if it applicable */
  TView with_layout(BNNSDataLayout layout) const {
    ICHECK_EQ(Tensor::getRank(view_desc_), Tensor::getRank(layout));

    TView res = *this;
    res.view_desc_.layout = layout;
    return res;
  }

  /** Construct party TView by splitting original TView into num parts */
  TView party_split_n(size_t num) const {
    ICHECK_EQ(party_size_, 1);

    TView res = *this;
    size_t rank = Tensor::getRank(view_desc_);
    size_t size = Tensor::getSize(view_desc_);
    res.party_size_ = num;
    res.party_stride_ = size / num;

    if (res.batch_size_ != 1) {
      res.batch_size_ /= num;
    } else {
      res.view_desc_.size[rank - 1] /= num;
      res.batch_stride_ /= num;
    }
    return res;
  }

  /** Construct party TView by duplicating original TView num times */
  TView party_duplicate_n(size_t num) const {
    ICHECK_EQ(party_size_, 1);

    TView res = *this;
    res.party_size_ = num;
    res.party_stride_ = 0;

    return res;
  }

  /** Return data buffer handler */
  HDL get_data_hdl() const { return view_desc_.data; }

  /** Return external batch dimension value */
  size_t get_batch_size() const { return batch_size_; }

  /** Return external batch dimension stride */
  size_t get_stride() const { return batch_stride_; }

  /** Return party element by index */
  TView operator[](size_t i) const {
    ICHECK_LT(i, party_size_);

    TView res = *this;
    res.party_size_ = 1;
    if (origin_) {
      auto hdl = reinterpret_cast<uint8_t*>(origin_->get_data_hdl());
      hdl += i * party_stride_ * Tensor::getElementSize(view_desc_.data_type);
      res.view_desc_.data = hdl;
    }
    return res;
  }

  /** Check if view is empty and doesn't relay to any tensor */
  operator bool() const { return origin_ != nullptr; }

  /** Get BNNS descriptor for particular View. Batch and Party attributed are ignored. */
  const BNNSNDArrayDescriptor& get_bnns_view() const { return view_desc_; }

 private:
  /** Original tensor object to view on */
  TensorPtr origin_;

  /** Batched view parameters */
  BNNSNDArrayDescriptor view_desc_ = {};
  size_t batch_size_ = 1;
  size_t batch_stride_ = 0;

  /** Party representation parameters */
  size_t party_size_ = 1;
  size_t party_stride_ = 0;
};

/**
 * Wrapper on top of BNNSFilter and src/dst TensorView.
 *
 * Support decomposed representation of filter and can execute sub primitives in parallel.
 */
class Primitive {
 public:
  Primitive(const std::vector<BNNSFilter> fs, const TView& src, const TView& dst)
      : filters(fs), src_view(src), dst_view(dst) {}

  virtual ~Primitive() {
    for (auto& filter : filters)
      if (filter) {
        BNNSFilterDestroy(filter);
        filter = nullptr;
      }
  }

  /** Execute primitive with using specified src/dst */
  void execute() {
    auto res = TVMBackendParallelLaunch(run_task, this, filters.size());
    ICHECK_EQ(res, 0) << "BNNS runtime. Primitive was not executed properly";
  }

 private:
  virtual int execute_impl(int part_idx) {
    const auto filter = this->filters[part_idx];
    const auto src_view = this->src_view[part_idx];
    const auto dst_view = this->dst_view[part_idx];

    size_t mb = src_view.get_batch_size();

    // NB! BNNS limitations
    //   * Do not use simple BNNSFilterApply. There is a bug inside BNNS,
    //     BNNSFilterApply doesn't work for grouped convolution.
    //   * Group convolution doesn't support arbitrary stride for Batch dim.
    //     The tensor should be dense.
    return BNNSFilterApplyBatch(filter, mb, src_view.get_data_hdl(), src_view.get_stride(),
                                dst_view.get_data_hdl(), dst_view.get_stride());
  }

  static int run_task(int task_id, TVMParallelGroupEnv* penv, void* cdata) {
    auto prim = reinterpret_cast<Primitive*>(cdata);
    return prim->execute_impl(task_id);
  }

 protected:
  /** BNNS kernels/filters collect which will execute primitive */
  std::vector<BNNSFilter> filters = {};
  const TView src_view;
  const TView dst_view;
};

/**
 * Wrapper on top of BNNS::Primitive
 *
 * This primitive should be used for executing primitive with two inputs.
 */
class TwoInputPrimitive : public Primitive {
 public:
  TwoInputPrimitive(const std::vector<BNNSFilter> fs, const TView& src, const TView& src2,
                    const TView& dst)
      : Primitive(fs, src, dst), src2_view(src2) {}

 private:
  int execute_impl(int task_id) override {
    const auto filter = this->filters[task_id];
    const auto src_view = this->src_view[task_id];
    const auto src2_view = this->src2_view[task_id];
    const auto dst_view = this->dst_view[task_id];

    size_t mb = src_view.get_batch_size();

    return BNNSFilterApplyTwoInputBatch(filter, mb, src_view.get_data_hdl(), src_view.get_stride(),
                                        src2_view.get_data_hdl(), src2_view.get_stride(),
                                        dst_view.get_data_hdl(), dst_view.get_stride());
  }

 protected:
  const TView src2_view;
};

/**
 * Wrapper on top of BNNS::Primitive
 *
 * This primitive should be used for executing normalization filter
 */
class NormPrimitive : public Primitive {
 public:
  using Primitive::Primitive;

 private:
  int execute_impl(int task_id) override {
    const auto filter = this->filters[task_id];
    const auto src_view = this->src_view[task_id];
    const auto dst_view = this->dst_view[task_id];

    size_t mb = src_view.get_batch_size();
    return BNNSNormalizationFilterApplyBatch(filter, mb, src_view.get_data_hdl(),
                                             src_view.get_stride(), dst_view.get_data_hdl(),
                                             dst_view.get_stride(), false);
  }
};

/**
 * Wrapper on top of BNNS::Primitive
 *
 * This primitive should be used for executing pooling filter
 */
class PoolingPrimitive : public Primitive {
 public:
  using Primitive::Primitive;

 private:
  int execute_impl(int task_id) override {
    const auto filter = this->filters[task_id];
    const auto src_view = this->src_view[task_id];
    const auto dst_view = this->dst_view[task_id];

    size_t mb = src_view.get_batch_size();
    return BNNSPoolingFilterApplyBatch(filter, mb, src_view.get_data_hdl(), src_view.get_stride(),
                                       dst_view.get_data_hdl(), dst_view.get_stride(), nullptr, 0);
  }
};

/**
 * Function which split primitive into sub primitives to parallel execution
 *
 * @param num requested num of sub primitives
 * @param orig_conv_param original convolution descriptor
 * @param src_view source tensor view
 * @param wgh_view weight tensor view
 * @param b_view bias tensor view
 * @param dst_view destination tensor view
 * @param num number of part to split into
 * @return collection of Convolution descriptors plus corresponding src/dst tensors view
 */
static std::tuple<std::vector<BNNSLayerParametersConvolution>, TView, TView> split_to_n(
    size_t num, const BNNSLayerParametersConvolution& orig_conv_param, const TView& src_view,
    const TView& wgh_view, const TView& b_view, const TView& dst_view) {
  size_t batch = src_view.get_batch_size();
  size_t oc = dst_view.get_bnns_view().size[2];
  size_t groups = orig_conv_param.groups;

  BNNS::TView src_view_new;
  BNNS::TView wgh_view_new;
  BNNS::TView b_view_new;
  BNNS::TView dst_view_new;

  // TODO(apeskov): Add split by batch dim. Meanwhile we just disable it...
  if (batch > 1 || oc % num != 0 || (groups > 1 && groups % num != 0)) {
    return {{orig_conv_param}, src_view, dst_view};
  }

  // if groups > 1 split only by groups
  // otherwise split inside one convolution by output channels
  if (groups > 1) {
    src_view_new = src_view.party_split_n(num);
    groups = groups / num;
  } else {
    src_view_new = src_view.party_duplicate_n(num);
  }

  wgh_view_new = wgh_view.party_split_n(num);
  b_view_new = b_view.party_split_n(num);
  dst_view_new = dst_view.party_split_n(num);

  std::vector<BNNSLayerParametersConvolution> res(num);
  for (size_t i = 0; i < num; i++) {
    auto& cur = res[i];
    cur = orig_conv_param;

    cur.i_desc = src_view_new[i].get_bnns_view();
    cur.o_desc = dst_view_new[i].get_bnns_view();
    cur.w_desc = wgh_view_new[i].get_bnns_view();
    cur.bias = b_view_new[i].get_bnns_view();
    cur.groups = groups;
  }
  return {res, src_view_new, dst_view_new};
}

}  // namespace BNNS
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_BNNS_BNNS_WRP_H_
