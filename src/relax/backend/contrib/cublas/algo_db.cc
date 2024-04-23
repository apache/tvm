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
 * \brief Implementation of codegen part of tuning capabilities for cublas matmul primitives.
 */

#include "algo_db.h"

#include <tvm/runtime/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

#include "../utils.h"
#include "../../../runtime/contrib/cublas/cublas_utils.h"
#include "../../../runtime/cuda/cuda_common.h"

#include <numeric>
#include <random>

namespace tvm {
namespace relax {
namespace contrib {

using namespace tvm::contrib;

const char * const matmulTileName(int idx) {
  const char * const names[] = {
    "UNDEF",
    "8x8",
    "8x16",
    "16x8",
    "8x32",
    "16x16",
    "32x8",
    "8x64",
    "16x32",
    "32x16",
    "64x8",
    "32x32",
    "32x64",
    "64x32",
    "32x128",
    "64x64",
    "128x32",
    "64x128",
    "128x64",
    "64x256",
    "128x128",
    "256x64",
    "64x512",
    "128x256",
    "256x128",
    "512x64",
  };
  if (idx >= 0 && idx < int(sizeof(names) / sizeof(names[0])))
    return names[idx];

  return "NA_OutOfRange";
}

const char * const clasterShapeName(int idx) {
  const char * const names[] = {
    "AUTO",
    "NA",
    "1x1x1",
    "2x1x1",
    "4x1x1",
    "1x2x1",
    "2x2x1",
    "4x2x1",
    "1x4x1",
    "2x4x1",
    "4x4x1",
    "8x1x1",
    "1x8x1",
    "8x2x1",
    "2x8x1",
    "16x1x1",
    "1x16x1",
    "3x1x1",
    "5x1x1",
    "6x1x1",
    "7x1x1",
    "9x1x1",
    "10x1x1",
    "11x1x1",
    "12x1x1",
    "13x1x1",
    "14x1x1",
    "15x1x1",
    "3x2x1",
    "5x2x1",
    "6x2x1",
    "7x2x1",
    "1x3x1",
    "2x3x1",
    "3x3x1",
    "4x3x1",
    "5x3x1",
    "3x4x1",
    "1x5x1",
    "2x5x1",
    "3x5x1",
    "1x6x1",
    "2x6x1",
    "1x7x1",
    "2x7x1",
    "1x9x1",
    "1x10x1",
    "1x11x1",
    "1x12x1",
    "1x13x1",
    "1x14x1",
    "1x15x1",
  };
  if (idx >= 0 && idx < int(sizeof(names) / sizeof(names[0])))
    return names[idx];

  return "NA_OutOfRange";
}

const char * innerShapeName(int idx) {
  const char * const names[] = {
    "UNDEFINED",
    "MMA884",
    "MMA1684",
    "MMA1688",
    "MMA16816",
  };

  if (idx >= 0 && idx < int(sizeof(names) / sizeof(names[0])))
    return names[idx];

  return "NA_OutOfRange";
}

const char * stagesName(int idx) {
  const char * const names[] = {
    "UNDEFINED",
    "16x1",
    "16x2",
    "16x3",
    "16x4",
    "16x5",
    "16x6",
    "32x1",
    "32x2",
    "32x3",
    "32x4",
    "32x5",
    "32x6",
    "64x1",
    "64x2",
    "64x3",
    "64x4",
    "64x5",
    "64x6",
    "128x1",
    "128x2",
    "128x3",
    "128x4",
    "128x5",
    "128x6",
    "32x10",
    "8x4",
    "16x10",
    "8x5",
    "NA_29",
    "NA_30",
    "8x3",
    "8xAUTO",
    "16xAUTO",
    "32xAUTO",
    "64xAUTO",
    "128xAUTO",
  };

  if (idx >= 0 && idx < int(sizeof(names) / sizeof(names[0])))
    return names[idx];

  return "NA_OutOfRange";
}

const char * const reductionSchemeName(int idx) {
  const char * const names[] = {
    "NONE",
    "INPLACE",
    "COMPUTE_TYPE",
    "INPLACE | COMPUTE_TYPE",
    "OUTPUT_TYPE",
    "OUTPUT_TYPE | INPLACE",
    "OUTPUT_TYPE | COMPUTE_TYPE",
    "OUTPUT_TYPE | INPLACE | COMPUTE_TYPE",
  };
  if (idx >= 0 && idx < int(sizeof(names) / sizeof(names[0])))
    return names[idx];

  return "NA_OutOfRange";
}

String to_string(const AlgoDesc& desc) {
  const cublasLtMatmulAlgo_t* algo = &desc->algo;
  int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stagesId;
  uint16_t inner_shape, claster_shape;

  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesId, sizeof(stagesId), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner_shape, sizeof(inner_shape), NULL));
  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigGetAttribute(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &claster_shape, sizeof(claster_shape), NULL));

  std::stringstream ss;
  ss << "id=" << algoId
    << " tile=" << matmulTileName(tile)
    << " splitK=" << numSplitsK
    << " reduc=" << reductionSchemeName(reductionScheme)
    << " swizzle=" << swizzle
    << " custom=" << customOption
    << " stagesId=" << stagesName(stagesId)
    << " innerShape=" << innerShapeName(inner_shape)
    << " clasterShape=" << clasterShapeName(claster_shape)
  ;
  return ss.str();
}

/********** AlgoDataBaseNode **********/

void AlgoDatabaseNode::Save(dmlc::JSONWriter* writer) const {
  std::vector<uint64_t> keys;
  std::vector<AlgoCollection> vals;
  keys.reserve(collections.size());
  vals.reserve(collections.size());
  for (auto& [key, val]: collections) {
    keys.push_back(key);
    vals.push_back(val);
  }

  writer->BeginObject();
  writer->WriteObjectKeyValue("collections_keys", keys);
  writer->WriteObjectKeyValue("collections_vals", vals);
  writer->EndObject();
}

void AlgoDatabaseNode::Load(dmlc::JSONReader* reader) {
  std::vector<uint64_t> keys;
  std::vector<AlgoCollection> vals;
  std::string key;
  reader->BeginObject();
  while (reader->NextObjectItem(&key)) {
    if (key == "collections_keys") {
      reader->Read(&keys);
    } else if (key == "collections_vals") {
      reader->Read(&vals);
    } else {
      LOG(FATAL) << "Unknown key '" << key << "'";
    }
  }
  ICHECK_EQ(keys.size(), vals.size());
  for (size_t i = 0; i < keys.size(); i++) {
    collections[keys[i]] = vals[i];
  }
}

String AlgoDatabaseNode::ToJSON() const {
  std::ostringstream os;
  dmlc::JSONWriter writer(&os);
  writer.Write(*this);
  return os.str();
}

void AlgoDatabaseNode::PutRec(uint64_t task_hash, const AlgoCollection algo_collection) {
  collections[task_hash] = algo_collection;
}

AlgoCollection AlgoDatabaseNode::FindRec(uint64_t task_hash) const {
  if (collections.count(task_hash) != 0)
    return collections.at(task_hash);
  
  return AlgoCollection{};
}

AlgoDatabase AlgoDatabase::FromJSON(String json) {
  AlgoDatabase db{};
  std::istringstream is(json);
  dmlc::JSONReader reader(&is);
  db->Load(&reader);
  return db;
}

AlgoDatabase::AlgoDatabase() {
  auto n = runtime::make_object<AlgoDatabaseNode>();
  data_ = std::move(n);
}

std::vector<AlgoDatabase>* ThreadLocalAlgoDatabases() {
  static thread_local std::vector<AlgoDatabase> tls;
  return &tls;
}

void AlgoDatabase::EnterWithScope() { ThreadLocalAlgoDatabases()->push_back(*this); }

void AlgoDatabase::ExitWithScope() { ThreadLocalAlgoDatabases()->pop_back(); }

Optional<AlgoDatabase> AlgoDatabase::Current() {
  auto tls = ThreadLocalAlgoDatabases();
  if (tls->empty()) {
    return NullOpt;
  } else {
    return tls->back();
  }
}

TVM_REGISTER_OBJECT_TYPE(AlgoDatabaseNode);
TVM_REGISTER_NODE_TYPE(AlgoDatabaseNode);

TVM_REGISTER_GLOBAL("relax.backend.contrib.AlgoDatabaseEnterWithScope").set_body_method(&AlgoDatabase::EnterWithScope);
TVM_REGISTER_GLOBAL("relax.backend.contrib.AlgoDatabaseExitWithScope").set_body_method(&AlgoDatabase::ExitWithScope);
TVM_REGISTER_GLOBAL("relax.backend.contrib.AlgoDatabaseFromJSON").set_body_typed(&AlgoDatabase::FromJSON);
TVM_REGISTER_GLOBAL("relax.backend.contrib.AlgoDatabaseToJSON").set_body_method<AlgoDatabase>(&AlgoDatabaseNode::ToJSON);
TVM_REGISTER_GLOBAL("relax.backend.contrib.AlgoDatabasePutRec").set_body_method<AlgoDatabase>(&AlgoDatabaseNode::PutRec);

/********** Tuning routine **********/

struct TaskDesc {
  int M, N, K;
  bool trans_a, trans_b;
  tvm::DataType dtype_a, dtype_b, dtype_c;
  tvm::DataType dtype_compute;

  static TaskDesc FromCompositeFunc (const Function& func) {
    std::array<int, 2> args_order = {0, 1};
    bool b_trans = false;
    if (func->attrs.GetAttr<String>("Composite").value() == "cublas.matmul_transposed") {
      args_order = {1, 0};
      b_trans = true;
    }

    auto extract_tensor_info = [](tvm::ObjectRef sinfo) {
      auto tsinfo = sinfo.as<tvm::relax::TensorStructInfoNode>();
      ICHECK(tsinfo);
      auto shape_expr = Downcast<ShapeExpr>(tsinfo->shape.value());
      auto shape = tvm::relax::backend::GetIntShape(shape_expr->values);

      return std::tuple(tsinfo->dtype, shape);
    };

    ICHECK(func->params.size() >= 2);
    auto [a_dtype, a_shape] = extract_tensor_info(func->params[args_order[0]]->struct_info_);
    auto [b_dtype, b_shape] = extract_tensor_info(func->params[args_order[1]]->struct_info_);
    auto [c_dtype, c_shape] = extract_tensor_info(func->ret_struct_info);

    // squeeze a_shape ND->2D, like [squeezed_M, K]
    auto squeeshed_m = std::accumulate(a_shape.begin(), a_shape.end() - 1, 1, std::multiplies());

    TaskDesc res;
    res.M = squeeshed_m;
    res.N = b_shape[b_trans ? 0 : 1];
    res.K = b_shape[b_trans ? 1 : 0];
    res.dtype_a = a_dtype;
    res.dtype_b = b_dtype;
    res.dtype_c = c_dtype;
    res.trans_a = false;
    res.trans_b = b_trans;

    // NOTE! Specific of cublas BYOC impl
    // This is correct for f16 and f32
    res.dtype_compute = c_dtype;

    return res;
  }
};

struct TuningConfig {
  size_t dyn_m_range = 128;
  size_t dyn_m_step = 1;
  size_t dyn_m_offset = 0;
  size_t num_trials = 1000;
  size_t num_repeats = 3;
  size_t repeat_size = 1000;
  float treshold_percent = 0.1;  // 10%;
  float merge_region_tolerance = 0.005;  // 0.5%
  bool heuristic_only = false;
  bool verbose = false;

  static TuningConfig FromMap(const Map<String, ObjectRef>& cfg_map) {
    TuningConfig res;
    for (auto& [k, v] : cfg_map) {
      if (k == "dyn_m_range") {
        res.dyn_m_range = Downcast<Integer>(v).IntValue();
      } else if (k == "dyn_m_step") {
        res.dyn_m_step = Downcast<Integer>(v).IntValue();
      } else if (k == "dyn_m_offset") {
        res.dyn_m_offset = Downcast<Integer>(v).IntValue();
      } else if (k == "num_trials") {
        res.num_trials = Downcast<Integer>(v).IntValue();
      } else if (k == "num_repeats") {
        res.num_repeats = Downcast<Integer>(v).IntValue();
      } else if (k == "repeat_size") {
        res.repeat_size = Downcast<Integer>(v).IntValue();
      } else if (k == "treshold_percent") {
        res.treshold_percent = Downcast<FloatImm>(v)->value;
      } else if (k == "merge_region_tolerance") {
        res.merge_region_tolerance = Downcast<FloatImm>(v)->value;
      } else if (k == "mode") {
        res.heuristic_only = Downcast<String>(v) == "cublas.heuristic_top1";
      } else if (k == "verbose") {
        res.verbose = Downcast<Bool>(v);
      } else {
        LOG(ERROR) << "Unknown key " << k;
      }
    }
    return res;
  }
};

static inline uint16_t to_f16(float val) {
  return __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(val);
}

struct TuningContext {
  TuningContext(TaskDesc tdesc, TuningConfig cfg) {
    CHECK_CUBLAS_ERROR(cublasLtCreate(&lth));
    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaEventCreate(&start_event));
    CUDA_CALL(cudaEventCreate(&stop_event));

    auto fill_normal = [&](void *dev_ptr, int size, const tvm::DataType& dtype) {
      auto host_arr = std::vector<uint8_t>(size * dtype.bytes());

      std::normal_distribution<float> normal_dist(0, 1);
      rnd_eng.seed(0);

      if (dtype == tvm::DataType::Float(16)) {
        std::generate_n(reinterpret_cast<uint16_t*>(host_arr.data()), size, [&]() {
                          return to_f16(normal_dist(rnd_eng));
                        });
      } else if (dtype == tvm::DataType::Float(32)) {
        std::generate_n(reinterpret_cast<float*>(host_arr.data()), size, [&]() {
                          return normal_dist(rnd_eng);
                        });
      } else {
        LOG(FATAL) << "Unsupported dtype " << dtype;
      }
      CUDA_CALL(cudaMemcpyAsync(dev_ptr, host_arr.data(), host_arr.size() * sizeof(host_arr[0]), cudaMemcpyHostToDevice, stream));
    };

    workspace_size = tvm::contrib::CuBlasLtThreadEntry::ThreadLocal()->workspace_size;;
    int max_m = cfg.dyn_m_range;

    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&Adev), tdesc.N * tdesc.K * tdesc.dtype_b.bytes()));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&Bdev), max_m * tdesc.K * tdesc.dtype_a.bytes()));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&Cdev), max_m * tdesc.N * tdesc.dtype_c.bytes()));
    CUDA_CALL(cudaMalloc(&workspace, workspace_size));

    fill_normal(Adev, tdesc.N * tdesc.K, tdesc.dtype_b);
    fill_normal(Bdev, max_m * tdesc.K, tdesc.dtype_a);
    fill_normal(Cdev, max_m * tdesc.N, tdesc.dtype_c);
  }
  
  ~TuningContext() {
    cudaFree(Adev);
    cudaFree(Bdev);
    cudaFree(Cdev);
    cudaFree(workspace);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cublasLtDestroy(lth);
  }

  void *Adev = nullptr;
  void *Bdev = nullptr;
  void *Cdev = nullptr;
  void *workspace = nullptr;

  size_t workspace_size;

  cudaStream_t stream  = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  
  cublasLtHandle_t lth = nullptr;

  std::mt19937 rnd_eng;
};


std::vector<AlgoDesc> tuneCublasAlgo(const TaskDesc& task_desc, const TuningConfig& cfg, const TuningContext& ctx) {
  const int64_t M = task_desc.N;
  const int64_t N = task_desc.M;
  const int64_t K = task_desc.K;
  
  auto a_type = GetCudaDataType(task_desc.dtype_b);
  auto b_type = GetCudaDataType(task_desc.dtype_a);
  auto c_type = GetCudaDataType(task_desc.dtype_c);
  auto compute_type = task_desc.dtype_compute == DataType::Float(32) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_16F;

  auto scale_type = CUDA_R_32F;
  float one_fp32 = 1.0;
  float zero_fp32 = 0.0;
  uint16_t one_fp16 = to_f16(1.0);
  uint16_t zero_fp16 = to_f16(0.0);
  void* alpha = &one_fp32;
  void* beta = &zero_fp32;

  if (c_type == CUDA_R_16F) {
    scale_type = CUDA_R_16F;
    alpha = &one_fp16;
    beta = &zero_fp16;
  }

  cublasLtMatmulDescOpaque_t op_desc_opaque;
  cublasLtMatmulDesc_t op_desc = &op_desc_opaque;
  cublasOperation_t op_transa = task_desc.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_transb = task_desc.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  CHECK_CUBLAS_ERROR(cublasLtMatmulDescInit(op_desc, compute_type, scale_type));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                    &op_transb, sizeof(op_transb)));
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                    &op_transa, sizeof(op_transa)));

  auto lda = !task_desc.trans_b ? M : K;
  auto ldb = !task_desc.trans_a ? K : N;
  auto ldc = M;

  cublasLtMatrixLayoutOpaque_t A_desc_opaque, B_desc_opaque, C_desc_opaque;
  cublasLtMatrixLayout_t A_desc = &A_desc_opaque;
  cublasLtMatrixLayout_t B_desc = &B_desc_opaque;
  cublasLtMatrixLayout_t C_desc = &C_desc_opaque;
  CHECK_CUBLAS_ERROR(
      cublasLtMatrixLayoutInit(A_desc, a_type, !task_desc.trans_b ? M : K, !task_desc.trans_b ? K : M, lda));
  CHECK_CUBLAS_ERROR(
      cublasLtMatrixLayoutInit(B_desc, b_type, !task_desc.trans_a ? K : N, !task_desc.trans_a ? N : K, ldb));
  CHECK_CUBLAS_ERROR(cublasLtMatrixLayoutInit(C_desc, c_type, M, N, ldc));

  // Get all available algo IDs. 
  std::vector<int> algo_ids(1);
  {
    int nb_algo_ids = 1;
    while (nb_algo_ids == (int)algo_ids.size()) {
      algo_ids.resize(algo_ids.size() * 2);
      CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetIds(ctx.lth, compute_type, scale_type, a_type, b_type, c_type, c_type, algo_ids.size(), algo_ids.data(), &nb_algo_ids));
    } 
    algo_ids.resize(nb_algo_ids);
  }

  // Construct search space
  std::vector<std::tuple<
    int, /* ID */
    std::vector<int>, /* TILE_IDS */
    std::vector<int>, /* STAGES_IDS */
    std::vector<int>, /* SPLITK_SUPPORT */
    std::vector<int>, /* REDUCTION_SCHEME_MASK */
    uint32_t, /* CTA_SWIZZLING_SUPPORT_MAX */
    int32_t  /* CUSTOM_OPTION_MAX */ 
  >> search_space;
  
  for (auto id : algo_ids) {
    cublasLtMatmulAlgo_t algo;
    size_t sizeWritten = 0;
    cublasStatus_t status = cublasLtMatmulAlgoInit(ctx.lth, compute_type, scale_type, a_type, b_type, c_type, c_type, id, &algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }

    std::vector<int> tile_ids;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten));
    if (sizeWritten) {
      tile_ids.resize(sizeWritten/sizeof(tile_ids[0]));
      CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tile_ids.data(), tile_ids.size() * sizeof(tile_ids[0]), &sizeWritten));
    } else {
      tile_ids = {CUBLASLT_MATMUL_TILE_UNDEFINED};
    }
    
    std::vector<int> stages_ids;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten));
    if (sizeWritten) {
      stages_ids.resize(sizeWritten/sizeof(stages_ids[0]));
      CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages_ids.data(), stages_ids.size() * sizeof(stages_ids[0]), &sizeWritten));
    } else {
      tile_ids = {CUBLASLT_MATMUL_TILE_UNDEFINED};
    }
    
    int32_t splitkSupport = 0;
    std::vector<int> splitk_vals = {-1, 1};
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten));
    if (sizeWritten && splitkSupport) {
      splitk_vals = {-1, 1, 2, 4, 8};  // some values which can be relevant for real cases of matmul appliance 
    }
    
    int32_t red_scheme_mask_support;
    std::vector<int32_t> red_scheme_vals = {CUBLASLT_REDUCTION_SCHEME_NONE};
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &red_scheme_mask_support, sizeof(red_scheme_mask_support), &sizeWritten));
    if (sizeWritten && red_scheme_mask_support) {
      for (auto mask : {CUBLASLT_REDUCTION_SCHEME_INPLACE, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE} ) {
        if ((red_scheme_mask_support & mask) == mask) {
          std::vector<int32_t> new_vals;
          for (auto cur_scheme : red_scheme_vals) {
            new_vals.push_back(cur_scheme || mask);
          }
          red_scheme_vals.insert(red_scheme_vals.end(), new_vals.begin(), new_vals.end());
        }
      }
    }
    uint32_t cta_swizzling_max;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &cta_swizzling_max, sizeof(cta_swizzling_max), &sizeWritten));
    
    int32_t custom_opt_max;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &custom_opt_max, sizeof(custom_opt_max), &sizeWritten));

    search_space.emplace_back(id, tile_ids, stages_ids, splitk_vals, red_scheme_vals, cta_swizzling_max, custom_opt_max);
  }

  auto benchmark_loop = [&](const cublasLtMatmulAlgo_t &algo, const int loop_size, const int num_loops) -> double {
    // warmup 
    CHECK_CUBLAS_ERROR(cublasLtMatmul(ctx.lth, op_desc, alpha,
                                  ctx.Adev, A_desc, ctx.Bdev, B_desc, beta,
                                  ctx.Cdev, C_desc, ctx.Cdev, C_desc,
                                  &algo, ctx.workspace, ctx.workspace_size,
                                  ctx.stream));

    std::vector<double> durs(num_loops);
    for (int loop_idx = 0; loop_idx < num_loops; loop_idx++) {
      CUDA_CALL(cudaEventRecord(ctx.start_event, ctx.stream));
      for (int loop = 0; loop < loop_size; loop++) {
          CHECK_CUBLAS_ERROR(cublasLtMatmul(ctx.lth, op_desc, alpha,
                                            ctx.Adev, A_desc, ctx.Bdev, B_desc, beta,
                                            ctx.Cdev, C_desc, ctx.Cdev, C_desc,
                                            &algo, ctx.workspace, ctx.workspace_size,
                                            ctx.stream));
      }
      CUDA_CALL(cudaEventRecord(ctx.stop_event, ctx.stream));
      CUDA_CALL(cudaEventSynchronize(ctx.stop_event));

      float time = 0;
      CUDA_CALL(cudaEventElapsedTime(&time, ctx.start_event, ctx.stop_event));
      double time_us = time / loop_size * 1000;
      durs[loop_idx] = time_us;
    }
    std::sort(durs.begin(), durs.end());
    return durs[num_loops / 2];  // median
  };

  auto benchmark = [&](const cublasLtMatmulAlgo_t &algo, const double time_us_limit) -> double {
    cublasLtMatmulHeuristicResult_t heur_res;
    cublasStatus_t status = cublasLtMatmulAlgoCheck(ctx.lth, op_desc, A_desc, B_desc, C_desc, C_desc, &algo, &heur_res);
    if (status != CUBLAS_STATUS_SUCCESS || heur_res.workspaceSize > ctx.workspace_size) {
        return 0;
    }

    auto time_us_preflight = benchmark_loop(algo, 1, 1);
    if (time_us_preflight > time_us_limit) 
      return 0;

    return benchmark_loop(algo, cfg.repeat_size, cfg.num_repeats);
  };

  std::vector<AlgoDesc> results = {};

  double ref_time_us = 0;
  {
    cublasLtMatmulPreferenceOpaque_t matmul_pref_opaque;
    cublasLtMatmulPreference_t matmul_pref_desc = &matmul_pref_opaque;

    cublasLtMatmulPreferenceInit(matmul_pref_desc);
    cublasLtMatmulPreferenceSetAttribute(matmul_pref_desc, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ctx.workspace_size, sizeof(size_t));

    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    int returned_result = 0;
    CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoGetHeuristic(ctx.lth, op_desc, A_desc, B_desc, C_desc, C_desc,
                                                      matmul_pref_desc, 1, &heuristic_result,
                                                      &returned_result));
    if (returned_result == 0) {
      CHECK_CUBLAS_ERROR(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    auto no_limit = std::numeric_limits<double>::infinity();
    ref_time_us = benchmark(heuristic_result.algo, no_limit);
    
    results.push_back(AlgoDesc{ref_time_us, heuristic_result.algo});

    if (cfg.verbose)
      LOG(INFO) << " M=" << N << " ref_us=" << ref_time_us;
  }

  if (cfg.heuristic_only)
    return results;

  // Main tuning loop
  for (auto space_elem : search_space) {
    auto algo_id = std::get<0>(space_elem);
    auto tile_ids = std::get<1>(space_elem);
    auto stages_ids = std::get<2>(space_elem);
    auto splitk_vals = std::get<3>(space_elem);
    auto red_scheme_vals = std::get<4>(space_elem);
    auto cta_swizzling_max = std::get<5>(space_elem);
    auto custom_opt_max = std::get<6>(space_elem);

    for (auto tile_id : tile_ids) {
      for (auto stages_id : stages_ids) {
        for (auto splitk : splitk_vals) {
          for (auto red_scheme : red_scheme_vals) {
            for (uint32_t cta_swizzling = 0; cta_swizzling < cta_swizzling_max; cta_swizzling++) {
              for (int32_t custom_opt = 0; custom_opt < custom_opt_max; custom_opt++) {
                for (uint16_t cluster_shape = 0; cluster_shape < CUBLASLT_CLUSTER_SHAPE_END; cluster_shape++) {
                  if (results.size() >= cfg.num_trials)
                    break;

                  cublasLtMatmulAlgo_t algo;
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoInit(ctx.lth, compute_type, scale_type, a_type, b_type, c_type, c_type, algo_id, &algo));

                  // Do not iterate through INNER_SHAPE values. It is strictly defined by implementation.
                  uint16_t algo_inner_shape = CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED;

                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(tile_id)));
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof(splitk)));
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red_scheme, sizeof(red_scheme)));
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &cta_swizzling, sizeof(cta_swizzling)));
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom_opt, sizeof(custom_opt)));
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_id, sizeof(stages_id)));
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &algo_inner_shape, sizeof(algo_inner_shape)));
                  CHECK_CUBLAS_ERROR(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster_shape, sizeof(cluster_shape)));

                  auto time_us = benchmark(algo, ref_time_us * (1 + cfg.treshold_percent));

                  if (time_us != 0.0) {
                    results.push_back(AlgoDesc{time_us, algo});
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (cfg.verbose) {
    double best = std::numeric_limits<double>::infinity();
    for (auto r : results) {
      best = r->estimated_time_us < best ? r->estimated_time_us : best;
    }
    LOG(INFO) << "best_us=" << best;
  }

  return results;
}

AlgoCollection AgregateToCollection(std::vector<std::pair<size_t, std::vector<AlgoDesc>>> raw_score) {
  AlgoCollection res;
  auto& ranges = res->regions;
  auto& algos = res->algos;

  auto find_best = [](const std::vector<AlgoDesc>& scores) -> AlgoDesc {
    auto best_it = std::min_element(scores.begin(), scores.end(), 
                                    [](auto a, auto best) { 
                                      return a->estimated_time_us < best->estimated_time_us; 
                                    });
    return *best_it;
  };

  auto identical = [](const AlgoDesc& a, const AlgoDesc& b) {
    auto& a_ = a->algo.data;
    auto& b_ = b->algo.data;
    return a_[0] == b_[0] && a_[1] == b_[1] && 
           a_[2] == b_[2] && a_[3] == b_[3] && 
           a_[4] == b_[4] && a_[5] == b_[5] && 
           a_[6] == b_[6] && a_[7] == b_[7];
  };

  for (auto& [dyn_m, scores] : raw_score) {
    if (scores.empty())
      continue;

    auto best = find_best(scores);
    if (ranges.empty()) {
      size_t algo_idx = algos.size();
      algos.emplace_back(best);
      ranges.emplace_back(dyn_m, algo_idx);
      continue;
    }
    
    AlgoDesc prev_algo = algos[ranges.back().second];

    if (identical(best, prev_algo)) {
      // Enhance previous region
      ranges.back().first = dyn_m;
    } else {
      // Put new region
      auto found = std::find_if(algos.begin(), algos.end(), [&](auto a) { return identical(a, best); }); 
      size_t algo_idx = std::distance(algos.begin(), found);
      if (found == algos.end()) {
        algos.emplace_back(best);
      }
      ranges.emplace_back(dyn_m, algo_idx);
    }
  }

  return res;
}

AlgoDatabase TuneCublasTasks(Array<Function> tasks, Map<String, ObjectRef> cfg_map) {
  auto cfg = TuningConfig::FromMap(cfg_map);
  AlgoDatabase db;
  StructuralHash hash;

  for (auto& task :tasks) {
    TaskDesc task_desc = TaskDesc::FromCompositeFunc(task);
    
    if (task_desc.M != -1 || task_desc.N == -1 || task_desc.K == -1) {
      LOG(WARNING) << "Algo tuning task doen't meet constrains. Only one dynamic shape var is supported.";
      continue;
    }

    TuningContext ctx(task_desc, cfg);
    std::vector<std::pair<size_t, std::vector<AlgoDesc>>> raw_perf;
    for (size_t dyn_m = cfg.dyn_m_offset; dyn_m <= cfg.dyn_m_range; dyn_m += cfg.dyn_m_step) {
      if (dyn_m == 0)
        continue;

      task_desc.M = dyn_m;  // Specify concrete dyn_m value for task desc
      auto score = tuneCublasAlgo(task_desc, cfg, ctx);
      raw_perf.emplace_back(dyn_m, score);
    }

    auto algo_colllection = AgregateToCollection(raw_perf);
    db->PutRec(hash(task), algo_colllection);
  }
  return db;
}

TVM_REGISTER_GLOBAL("contrib.cublas.TuneAlgoTasks").set_body_typed(&TuneCublasTasks);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
