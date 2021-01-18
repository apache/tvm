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
 * \brief Simple JSON runtime for Apple BNNS primitives
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_backend_api.h>

#include <cstddef>
#include <string>
#include <vector>
#include <numeric>
#include <unordered_map>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "Accelerate/Accelerate.h"

#define BNNS_TMP_CONCURRENCY 2
#define BNNS_MAX_CONCURRENCY 8

template<typename T1, typename T2>
bool one_of(T1 arg1, T2 arg2) {
  return arg1 == arg2;
}

template<typename T1, typename T2, typename ...T>
bool one_of(T1 arg1, T2 arg2, T... args) {
  return arg1 == arg2 || one_of(arg1, args...);
}

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

/** C++ wrapper on top of original BNNS C api */
namespace BNNS {
using Dim = size_t;
using Shape = std::vector<Dim>;
using Dtype = BNNSDataType;

void* default_alloc(size_t size) {
  // TODO(apeskov): Clarify, should it have some alignment for better performance
  //   with SIMD execution.. may be TVMBackendAllocWorkspace is more preferable here.
  //   Note: Apple uses posix_memalign by default.
  return malloc(size);
}

void default_free(void* ptr) {
  free(ptr);
}

class Tensor {
 public:
  Tensor(Shape shape, Dtype dtype, void* hdl)
      : real_shape(shape) {
    ICHECK(shape.size() < BNNS_MAX_TENSOR_DIMENSION);

    if (hdl) {
      data_handler = hdl;
      is_external_data = true;
    } else {
      const size_t buff_size = getNumOfElements(shape) * getElementSize(dtype);
      data_handler = default_alloc(buff_size);
      is_external_data = false;
    }

    bnns_nd_desc = {
      BNNSNDArrayFlags(0),
      getPlainLayout(shape),
      {},       // shape
      {},       // strides, empty value means use default dense strides
      hdl,      // data handler
      dtype,    // data type
      nullptr,  // table_data (clustering case), is not used
      dtype,
      1.f,
      0.f
    };
    std::copy(shape.rbegin(), shape.rend(), std::begin(bnns_nd_desc.size));
  }

  ~Tensor() {
    if (data_handler && !is_external_data) {
      default_free(data_handler);
      data_handler = nullptr;
    }
  }

  void* get_data_hdl() { return data_handler; }

  const void* get_data_hdl() const { return data_handler; }

  void set_data_hdl(void *hdl) {
    if (data_handler && !is_external_data) {
      default_free(data_handler);
      data_handler = nullptr;
    }

    data_handler = hdl;
    is_external_data = true;
  }

  size_t get_mb() const {
    return real_shape[0];
  }

  size_t get_mb_stride() const {
    return std::accumulate(real_shape.begin() + 1, real_shape.end(),
                           1, std::multiplies<int>());
  }

  const BNNSNDArrayDescriptor get_nd_desc(size_t nd = 0) const {
    auto original_nd = real_shape.size();
    // Ask of original descriptor
    if (original_nd == nd || nd == 0)
      return bnns_nd_desc;

    // As of desc with excluded batch
    if (original_nd == nd + 1) {
      auto res = bnns_nd_desc;
      res.size[original_nd - 1] = 0;
      res.layout = BNNSDataLayout3DLastMajor;  // TODO(apeskov): hardcoded value. FIXME
      return res;
    }
    LOG(FATAL) << "Unknown case of BNNS tensor interpretation";
    return bnns_nd_desc;
  }

 private:
  static BNNSDataLayout getPlainLayout(const Shape &shape) {
    return getPlainLayout(shape.size());
  }

  static BNNSDataLayout getPlainLayout(size_t rank) {
    switch (rank) {
      case 1: return BNNSDataLayout1DFirstMajor;
      case 2: return BNNSDataLayout2DFirstMajor;
      case 3: return BNNSDataLayout3DFirstMajor;
      case 4: return BNNSDataLayout4DFirstMajor;
      case 5: return BNNSDataLayout5DFirstMajor;
      case 6: return BNNSDataLayout6DFirstMajor;
      case 7: return BNNSDataLayout7DFirstMajor;
      case 8: return BNNSDataLayout8DFirstMajor;
      default:
        LOG(FATAL) << "Unsupported tensor rank : " << rank
                   << " Supported cases is only 1-8 ";
        return static_cast<BNNSDataLayout>(0);
    }
  }

  /**
   * return size in byte of element of provided type
   * @param dtype
   * @return size of element in bytes
   */
  static size_t getElementSize(Dtype dtype) {
    return (dtype & 0xFFFF) / sizeof(uint8_t);
  }

  static size_t getNumOfElements(const Shape &shape) {
    return std::accumulate(shape.begin(), shape.end(),
                           1, std::multiplies<int>());
  }

 private:
  Shape  real_shape = {};
  void*  data_handler = nullptr;
  bool   is_external_data = false;

  BNNSNDArrayDescriptor bnns_nd_desc;
};

class Primitive {
public:
  explicit Primitive(BNNSFilter f) : num_filters(1), filters{f} {}

  explicit Primitive(BNNSFilter fs[BNNS_MAX_CONCURRENCY]) {
    std::copy(fs, fs + BNNS_MAX_CONCURRENCY, filters);
    for (int i = 0; i < BNNS_MAX_CONCURRENCY; i++) {
      if (filters[i] == nullptr) {
        num_filters = i;
        break;
      }
    }
  }

  ~Primitive() {
    for (size_t i = 0; i < num_filters; i++) {
      auto &filter = filters[i];
      if (filter) {
        BNNSFilterDestroy(filter);
        filter = nullptr;
      }
    }
  }

  void execute(std::vector<Tensor*> srcs, Tensor *dst, int forceBatchSize = -1) {
    ICHECK_LE(srcs.size(), 2) << "Currently BNNS runtime supports primitives with only 1 or 2 "
                                 "data inputs.";

    run_ctx ctx { this, srcs[0], nullptr, dst, forceBatchSize };
    if (srcs.size() > 1)
      ctx.src2 = srcs[1];

    auto res = TVMBackendParallelLaunch(run_task, &ctx, num_filters);
    ICHECK_EQ(res, 0) << "BNNS runtime. Primitive was not executed properly";
  }

  void set_input_stride(size_t stride1, size_t stride2 = 0) {
    in1_hdl_stride = stride1;
    in2_hdl_stride = stride2;
  }
  void set_output_stride(size_t stride) { out_hdl_stride = stride; }

 private:
  struct run_ctx {
    Primitive *prim;
    const Tensor *src1;
    const Tensor *src2;
    Tensor *dst;
    const int force_batch_size;
  };

  static int run_task(int task_id, TVMParallelGroupEnv* penv, void* cdata) {
    auto ctx = reinterpret_cast<run_ctx*>(cdata);
    const auto *prim = ctx->prim;

    const auto &filter  = prim->filters[task_id];

    auto src1_hdl = ctx->src1->get_data_hdl();
    auto dst_hdl  = ctx->dst->get_data_hdl();

    auto src1_mb = ctx->src1->get_mb();
    auto dst_mb = ctx->dst->get_mb();

    auto src1_mb_stride = ctx->src1->get_mb_stride();
    auto dst_mb_stride = ctx->dst->get_mb_stride();

    src1_hdl = static_cast<const uint8_t*>(src1_hdl) + task_id*prim->in1_hdl_stride;
    dst_hdl = static_cast<uint8_t*>(dst_hdl) + task_id*prim->out_hdl_stride;

    ICHECK(src1_mb == dst_mb) << "Mismatch of batch dimension of input/output tensors";

    const void* src2_hdl = nullptr;
    size_t src2_mb = 0;
    size_t src2_mb_stride = 0;

    if (ctx->src2) {
      src2_hdl  = ctx->src2->get_data_hdl();
      src2_mb = ctx->src2->get_mb();
      src2_mb_stride = ctx->src2->get_mb_stride();
      src2_hdl = static_cast<const uint8_t*>(src2_hdl) + task_id*prim->in2_hdl_stride;
      ICHECK(src2_mb == dst_mb) << "Mismatch of batch dimension of input/output tensors";
    }

    const auto mb = (ctx->force_batch_size == -1) ? dst_mb : ctx->force_batch_size;

    // WA
    if (mb == 1) {
      src1_mb_stride = prim->in1_hdl_stride / sizeof(float);
      dst_mb_stride = prim->out_hdl_stride / sizeof(float);
    }

    // NB! Limitations
    //   * Do not use simple BNNSFilterApply. There is a bug inside BNNS,
    //     and BNNSFilterApply doesn't work for grouped convolution.
    //   * Group convolution doesn't support arbitrary stride for Batch dim.
    //     The tensor should be dense.
    auto sts = (ctx->src2)
        ? BNNSFilterApplyTwoInputBatch(filter, mb,
              src1_hdl, src1_mb_stride,
              src2_hdl, src2_mb_stride,
              dst_hdl, dst_mb_stride)
        : BNNSFilterApplyBatch(filter, mb,
              src1_hdl, src1_mb_stride,
              dst_hdl, dst_mb_stride);

    return sts;
  }

 private:
  size_t num_filters = 0;
  BNNSFilter filters[BNNS_MAX_CONCURRENCY] = {};

  // TODO(apeskov): temporal solution with strides
  size_t in1_hdl_stride = 0;
  size_t in2_hdl_stride = 0;
  size_t out_hdl_stride = 0;
};

}  // namespace BNNS

struct BNNSThreadingConfig {
  /**
   * Internal parallelism level ov BNNS primitive specified via parameter.
   * Has no real control from TVM level, so in fact it may be ignored by
   * implementation.
   */
  int internalConcurrency = 0;

  /**
   * TVM level parallelism for BNNS primitive. In case if BNNS doesn't support
   * internal parallelism we can add it by splitting primitive into independent
   * parts and run it in parallel. May provide additional performance.
   */
  int externalConcurrency = 0;
};

class BNNSJSONRuntime : public JSONRuntimeBase {
 public:
  BNNSJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const override { return "bnns_json"; }

  void Init(const Array<NDArray>& consts) override {
    SetupConstants(consts);
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
  }

  void Run() override {
    // Wrap external handler into BNNS tensor representation
    auto bind_ext_hdl_to_tensor = [this] (uint32_t eid) {
      const auto &ext_dlt = *data_entry_[eid];
      auto &bnns_tensor = *entry_out_mem_[eid];
      bnns_tensor.set_data_hdl(ext_dlt.data);
    };

    // Bind all input/output external data object into internal abstractions
    for (const auto &eid : input_var_eid_) {
      bind_ext_hdl_to_tensor(eid);
    }
    for (const auto &out_entity : outputs_) {
      bind_ext_hdl_to_tensor(EntryID(out_entity));
    }

    // Invoke primitives in topological order
    for (int i = 0; i < primitives_.size(); ++i) {
      auto res = entry_out_mem_.at(prim_results_[i]);
      std::vector<BNNS::Tensor*> args;
      for (auto arg_id : prim_args_[i])
        args.push_back(entry_out_mem_.at(arg_id).get());

      int forceBatchSize =
          (force_batch_size_.find(i) == force_batch_size_.end()) ? -1 : force_batch_size_.at(i);
      primitives_.at(i)->execute(args, res.get(), forceBatchSize);
    }
  }

 private:
  // Build up the engine based on the input graph.
  void BuildEngine() {
    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("bnns.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("bnns.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("bnns.conv2d_bias" == op_name) {
          Conv2d(nid, false, true);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("bnns.dense_bias" == op_name) {
          Dense(nid, true);
        } else if ("bnns.dense_bias_gelu" == op_name) {
          Dense(nid, true, true);
        } else if ("nn.batch_matmul" == op_name) {
          MatMul(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a BNNS tensor.
  std::shared_ptr<BNNS::Tensor> BindBNNSTensor(const JSONGraphNodeEntry& entry,
                                               void *hdl = nullptr) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      auto data_node = nodes_[entry.id_];
      auto dlshape = data_node.GetOpShape()[entry.index_];
      auto dltype = data_node.GetOpDataType()[entry.index_];

      entry_out_mem_[eid] = std::make_shared<BNNS::Tensor>(
          BNNS::Shape{dlshape.begin(), dlshape.end()},
          convertToBNNS(dltype), hdl);
    }
    return entry_out_mem_[eid];
  }

  /**
   * Function which split primitive into sub primitives to parallel execution
   *
   * @param orig_conv_param descriptor of original convolution
   * @param batch batch value
   * @param num number of part to split into.
   * @return collection of Convolution descriptors plus strides for input and output tensors
   */
  static std::tuple<std::vector<BNNSLayerParametersConvolution>, size_t, size_t>
      split_into_n(const BNNSLayerParametersConvolution& orig_conv_param,
               size_t batch, size_t num) {
    size_t i_shape[BNNS_MAX_TENSOR_DIMENSION] = {};
    size_t o_shape[BNNS_MAX_TENSOR_DIMENSION] = {};
    size_t w_shape[BNNS_MAX_TENSOR_DIMENSION] = {};
    size_t b_shape[BNNS_MAX_TENSOR_DIMENSION] = {};
    size_t w_stride = 0;
    size_t b_stride = 0;
    size_t i_stride = 0;
    size_t o_stride = 0;

    // TODO(apeskov): In case of batch we can split through bach dimension.
    //   Meanwhile we just disable it...
    if (batch > 1) {
      return {{orig_conv_param}, 0, 0};
    }

    auto groups = orig_conv_param.groups;
    // if groups > 1 split only by groups
    // otherwise split inside one convolution by output channels
    if (groups > 1) {
      // fallback into sequential execution
      if (groups % num != 0)
        return {{orig_conv_param}, 0, 0};

      std::copy(orig_conv_param.i_desc.size, orig_conv_param.i_desc.size + 3, i_shape);
      std::copy(orig_conv_param.o_desc.size, orig_conv_param.o_desc.size + 3, o_shape);
      std::copy(orig_conv_param.w_desc.size, orig_conv_param.w_desc.size + 4, w_shape);
      std::copy(orig_conv_param.bias.size, orig_conv_param.bias.size + 1, b_shape);

      auto orig_w_buff_size = std::accumulate(w_shape, w_shape + 4, 1, std::multiplies<int>())
                              * sizeof(float);

      auto orig_b_buff_size = std::accumulate(b_shape, b_shape + 1, 1, std::multiplies<int>())
                              * sizeof(float);

      auto orig_i_buff_size = std::accumulate(i_shape, i_shape + 3, 1, std::multiplies<int>())
                              * sizeof(float);

      auto orig_o_buff_size = std::accumulate(o_shape, o_shape + 3, 1, std::multiplies<int>())
                              * sizeof(float);

      i_shape[2] /= num;
      o_shape[2] /= num;
      w_shape[3] /= num;
      b_shape[0] /= num;

      w_stride = orig_w_buff_size / num;
      b_stride = orig_b_buff_size / num;
      i_stride = orig_i_buff_size / num;
      o_stride = orig_o_buff_size / num;
      groups = groups / num;
    } else {
      std::copy(orig_conv_param.i_desc.size, orig_conv_param.i_desc.size + 3, i_shape);
      std::copy(orig_conv_param.o_desc.size, orig_conv_param.o_desc.size + 3, o_shape);
      std::copy(orig_conv_param.w_desc.size, orig_conv_param.w_desc.size + 4, w_shape);
      std::copy(orig_conv_param.bias.size, orig_conv_param.bias.size + 1, b_shape);

      auto orig_w_buff_size = std::accumulate(w_shape, w_shape + 4, 1, std::multiplies<int>())
                              * sizeof(float);

      auto orig_b_buff_size = std::accumulate(b_shape, b_shape + 1, 1, std::multiplies<int>())
                              * sizeof(float);

//      auto orig_i_buff_size = std::accumulate(i_shape, i_shape + 3, 1, std::multiplies<int>())
//                              * sizeof(float);

      auto orig_o_buff_size = std::accumulate(o_shape, o_shape + 3, 1, std::multiplies<int>())
                              * sizeof(float);

      o_shape[2] /= num;
      w_shape[3] /= num;
      b_shape[0] /= num;

      w_stride = orig_w_buff_size / num;
      b_stride = orig_b_buff_size / num;
      i_stride = 0;
      o_stride = orig_o_buff_size / num;
    }

    std::vector<BNNSLayerParametersConvolution> res(num);
    for (size_t i=0; i < num; i++) {
      auto &cur = res[i];
      cur = orig_conv_param;

      std::copy(i_shape, i_shape + 3, cur.i_desc.size);
      std::copy(o_shape, o_shape + 3, cur.o_desc.size);
      std::copy(w_shape, w_shape + 4, cur.w_desc.size);
      std::copy(b_shape, b_shape + 1, cur.bias.size);

      cur.w_desc.data = static_cast<uint8_t*>(cur.w_desc.data) + w_stride * i;
      if (cur.bias.data)
        cur.bias.data = static_cast<uint8_t*>(cur.bias.data) + b_stride * i;

      cur.groups = groups;
    }
    return {res, i_stride, o_stride};
  }


  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    auto dl_input_shape = nodes_[src_entry.id_].GetOpShape()[src_entry.index_];
    auto dl_weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    BNNS::Shape input_shape {dl_input_shape.begin(), dl_input_shape.end()};
    BNNS::Shape weight_shape {dl_weight_shape.begin(), dl_weight_shape.end()};
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilation = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    BNNS::Dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    BNNS::Dim N = input_shape[0],               // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[0]),       // height padding: left
        PH_R = std::stoi(str_padding[2]),       // height padding: right
        PW_L = std::stoi(str_padding[1]),       // width padding: left
        PW_R = std::stoi(str_padding[3]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[1]),         // weight-wise stride
        DH = std::stoi(str_dilation[0]),        // height kernel dilation
        DW = std::stoi(str_dilation[1]),        // width kernel dilation
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    BNNS::Shape src_dims = {N, IC, IH, IW};
    BNNS::Shape weights_dims = {OC, IC, KH, KW};
    BNNS::Shape bias_dims = {OC};
    BNNS::Shape dst_dims = {N, OC, OH, OW};
    BNNS::Shape strides_dims = {SH, SW};
    BNNS::Shape padding_dims_l = {PH_L, PW_L};
    BNNS::Shape padding_dims_r = {PH_R, PW_R};

    auto weight_data_entry = data_entry_[EntryID(weight_entry)];
    ICHECK(weight_data_entry) << "Convolution weights tensor should be constant and "
                                 "available on initialization stage. Looks like weights "
                                 "are not result of constant expression.";

    auto weight_ext_data_hdl = weight_data_entry->data;

    // Memory descriptions.
    auto src_md = BindBNNSTensor(src_entry);
    auto weights_md = BindBNNSTensor(weight_entry, weight_ext_data_hdl);
    std::shared_ptr<BNNS::Tensor> bias_md;
    auto dst_md = BindBNNSTensor(dst_entry);
    // TODO(apeskov): check correctness of tensor shapes

    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      auto bias_data_entry = data_entry_[EntryID(bias_entry)];
      ICHECK(bias_data_entry) << "Convolution bias tensor should be constant and "
                                 "available on initialization stage. Looks like bias "
                                 "is not result of constant expression.";

      auto bias_data_hdl = bias_data_entry->data;
      bias_md = BindBNNSTensor(bias_entry, bias_data_hdl);
    } else {
      bias_md = std::make_shared<BNNS::Tensor>(BNNS::Shape {OC}, BNNSDataTypeFloat32, nullptr);
    }

    BNNSActivation activation = { has_relu ?
        BNNSActivationFunctionRectifiedLinear :
        BNNSActivationFunctionIdentity };

    auto src_candidate = src_md->get_nd_desc(3);
    auto weights_candidate = weights_md->get_nd_desc();
    auto dst_candidate = dst_md->get_nd_desc(3);
    auto bias_candidate = bias_md->get_nd_desc();
    src_candidate.layout = BNNSDataLayoutImageCHW;
    dst_candidate.layout = BNNSDataLayoutImageCHW;
    weights_candidate.layout = BNNSDataLayoutConvolutionWeightsOIHW;
    bias_candidate.layout = BNNSDataLayoutVector;

    // TODO(apeskov): Tmp WA, broadcast bias is here with tailing [1, 1]
    if (bias_candidate.size[0] == 1 && bias_candidate.size[1] == 1 &&
        one_of(bias_candidate.size[3], 1, 0) &&
        std::all_of(bias_candidate.size + 4, bias_candidate.size + BNNS_MAX_TENSOR_DIMENSION,
            [] ( size_t d) { return d == 0; })) {
      auto element_count = bias_candidate.size[2];
      std::fill(bias_candidate.size, bias_candidate.size + BNNS_MAX_TENSOR_DIMENSION, 0);
      bias_candidate.size[0] = element_count;
    }

    BNNSLayerParametersConvolution conv_param = {
        src_candidate,
        weights_candidate,
        dst_candidate,
        bias_candidate,
        activation,
        SW, /* x_stride */
        SH, /* y_stride */
        DW, /* x_dilation_stride */
        DH, /* y_dilation_stride */
        0,  /* x_padding, explicit pads will be used */
        0,  /* y_padding, explicit pads will be used */
        groups, /* groups */
        {PW_L, PW_R, PH_L, PH_R} /* explicit pad values */
    };

    BNNSFilter filters[BNNS_MAX_CONCURRENCY] = {};

    std::vector<BNNSLayerParametersConvolution> params;
    size_t i_stride, o_stride;
    std::tie(params, i_stride, o_stride) = split_into_n(conv_param, N, BNNS_TMP_CONCURRENCY);
    for (int i = 0; i < params.size(); i++) {
      filters[i] = BNNSFilterCreateLayerConvolution(&params[i], &common_filter_param);
      ICHECK(filters[i]) << "BNNS primitive was not created. Unsupported attributes configuration";
    }

    primitives_.emplace_back(std::make_shared<BNNS::Primitive>(filters));
    primitives_.back()->set_input_stride(i_stride);
    primitives_.back()->set_output_stride(o_stride);

    prim_args_.push_back({EntryID(src_entry)});
    prim_results_.push_back({EntryID(dst_entry)});
  }

  void Dense(const size_t& nid, const bool has_bias = false, const bool has_gelu = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto src_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);

    auto w_data = data_entry_[EntryID(weight_entry)]->data;
    // Memory descriptions.
    auto src_md = BindBNNSTensor(src_entry);
    auto weights_md = BindBNNSTensor(weight_entry, w_data);
    auto dst_md = BindBNNSTensor(dst_entry);

    BNNSNDArrayDescriptor in_desc = src_md->get_nd_desc(1);
    BNNSNDArrayDescriptor w_desc = weights_md->get_nd_desc(2);
    BNNSNDArrayDescriptor out_desc = dst_md->get_nd_desc(1);
    w_desc.layout = BNNSDataLayoutRowMajorMatrix;
    in_desc.layout = BNNSDataLayoutVector;
    out_desc.layout = BNNSDataLayoutVector;
    w_desc.data = w_data;
    BNNSNDArrayDescriptor bias = {};
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      auto bias_data = data_entry_[EntryID(bias_entry)]->data;
      auto bias_md = BindBNNSTensor(bias_entry, bias_data);
      bias = bias_md->get_nd_desc();
      bias.layout = BNNSDataLayoutVector;
      bias.data = bias_data;
    }
    BNNSActivation activation = {BNNSActivationFunctionIdentity};
    if (has_gelu) {
        activation = {BNNSActivationFunctionGELUApproximation};
        activation.alpha = std::sqrt(2.0 / M_PI);
        activation.beta = 0.044715;
    }

    BNNSLayerParametersFullyConnected layerParameters = {
        in_desc,
        w_desc,
        out_desc,
        bias,
        activation,
    };

    auto filter = BNNSFilterCreateLayerFullyConnected(&layerParameters, &common_filter_param);
    ICHECK(filter) << "BNNS primitive was not created. Unsupported attributes configuration";
    primitives_.emplace_back(std::make_shared<BNNS::Primitive>(filter));
    prim_args_.push_back({EntryID(src_entry)});
    prim_results_.push_back({EntryID(dst_entry)});
  }

  void MatMul(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto a_entry = node.GetInputs()[0];
    auto b_entry = node.GetInputs()[1];
    auto dst_entry = JSONGraphNodeEntry(nid, 0);
    bool a_is_weighted = data_entry_[EntryID(a_entry)] != nullptr;
    bool b_is_weighted = data_entry_[EntryID(b_entry)] != nullptr;

    void* a_data = nullptr;
    void* b_data = nullptr;
    if (a_is_weighted)
        a_data = data_entry_[EntryID(a_entry)]->data;
    if (b_is_weighted)
        b_data = data_entry_[EntryID(b_entry)]->data;
    // Memory descriptions.
    auto a_md = BindBNNSTensor(a_entry, a_data);
    auto b_md = BindBNNSTensor(b_entry, b_data);
    auto dst_md = BindBNNSTensor(dst_entry);

    BNNSNDArrayDescriptor a_desc = a_md->get_nd_desc();
    BNNSNDArrayDescriptor b_desc = b_md->get_nd_desc();
    BNNSNDArrayDescriptor out_desc = dst_md->get_nd_desc();
    std::reverse(a_desc.size, a_desc.size + 3);
    std::reverse(b_desc.size, b_desc.size + 3);
    std::reverse(out_desc.size, out_desc.size + 3);
    a_desc.data = a_data;
    b_desc.data = b_data;

    BNNSLayerParametersBroadcastMatMul layerParameters = {
        1,  // alpha
        0,  // beta
        false,  // transA
        true,   // transB
        false,  // quadratic
        a_is_weighted,
        b_is_weighted,
        a_desc,
        b_desc,
        out_desc
    };

    auto filter = BNNSFilterCreateLayerBroadcastMatMul(&layerParameters, &common_filter_param);
    ICHECK(filter) << "BNNS primitive was not created. Unsupported attributes configuration";
    primitives_.emplace_back(std::make_shared<BNNS::Primitive>(filter));
    std::vector<uint32_t> args;
    if (!a_is_weighted)
        args.push_back(EntryID(a_entry));
    if (!b_is_weighted)
        args.push_back(EntryID(b_entry));
    prim_args_.push_back(std::move(args));
    prim_results_.push_back(EntryID(dst_entry));
    force_batch_size_.insert({prim_args_.size() - 1, 1});
  }

  BNNS::Dtype convertToBNNS(const DLDataType &dl_dtype) {
    if (dl_dtype.code == DLDataTypeCode::kDLFloat) {
      if (dl_dtype.bits == 32) return BNNSDataTypeFloat32;
      if (dl_dtype.bits == 16) return BNNSDataTypeFloat16;
    }
    if (dl_dtype.code == DLDataTypeCode::kDLInt) {
      if (dl_dtype.bits == 32) return BNNSDataTypeInt32;
      if (dl_dtype.bits == 16) return BNNSDataTypeInt16;
      if (dl_dtype.bits == 8) return BNNSDataTypeInt8;
    }
    if (dl_dtype.code == DLDataTypeCode::kDLUInt) {
      if (dl_dtype.bits == 32) return BNNSDataTypeUInt32;
      if (dl_dtype.bits == 16) return BNNSDataTypeUInt16;
      if (dl_dtype.bits == 8) return BNNSDataTypeUInt8;
    }
    LOG(FATAL) << "Unsupported data type for BNNS runtime";
    return BNNS::Dtype(0);
  }

  // TODO(apeskov): Allow to specify num of threads and keep customer buffers.
  //                Should investigate this attributes.
  BNNSFilterParameters common_filter_param {};

  std::vector<std::shared_ptr<BNNS::Primitive>> primitives_;
  std::vector<std::vector<uint32_t>> prim_args_;
  std::vector<uint32_t> prim_results_;
  std::unordered_map<uint32_t, uint32_t> force_batch_size_;

  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::shared_ptr<BNNS::Tensor>> entry_out_mem_;
};

runtime::Module BNNSJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<BNNSJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.BNNSJSONRuntimeCreate")
    .set_body_typed(BNNSJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_bnns_json")
    .set_body_typed(BNNSJSONRuntime::LoadFromBinary<BNNSJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
