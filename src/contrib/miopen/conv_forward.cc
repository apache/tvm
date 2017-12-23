/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external miopen utils function
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include "miopen_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;


TVM_REGISTER_GLOBAL("tvm.contrib.miopen.conv2d.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  const int mode = args[0];
  const int algo = args[1];
  const int pad_h = args[2];
  const int pad_w = args[3];
  const int stride_h = args[4];
  const int stride_w = args[5];
  const int dilation_h = args[6];
  const int dilation_w = args[7];
  const DLTensor *x = args[8];
  const DLTensor *w = args[9];
  const DLTensor *y = args[10];

  MIOpenThreadEntry* entry_ptr = MIOpenThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<miopenConvolutionMode_t>(mode);
  // Set Ctx
  entry_ptr->conv_entry.ctx = x->ctx;
  // Set Data Type
  entry_ptr->conv_entry.data_type = miopenFloat; // MIOpen only suppports fp32
  // Set Desc
  MIOPEN_CALL(miopenInitConvolutionDescriptor(entry_ptr->conv_entry.conv_desc,
					     entry_ptr->conv_entry.mode,
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w
                                             ));

  // Set Filter
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.filter_desc,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(w->shape[0]),
                                        static_cast<int>(w->shape[1]),
                                        static_cast<int>(w->shape[2]),
                                        static_cast<int>(w->shape[3])));
  // Set Input
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.input_desc,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(x->shape[0]),
                                        static_cast<int>(x->shape[1]),
                                        static_cast<int>(x->shape[2]),
                                        static_cast<int>(x->shape[3])));
  // Set Output
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.output_desc,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(y->shape[0]),
                                        static_cast<int>(y->shape[1]),
                                        static_cast<int>(y->shape[2]),
                                        static_cast<int>(y->shape[3])));
  // Set workspace
  size_t workspace_size = 0;
  MIOPEN_CALL(miopenConvolutionForwardGetWorkSpaceSize(entry_ptr->handle,
						       entry_ptr->conv_entry.filter_desc,
                                                     entry_ptr->conv_entry.input_desc,
                                                     entry_ptr->conv_entry.conv_desc,
                                                     entry_ptr->conv_entry.output_desc,
                                                     &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);

  const int request_algo_count = 4;
  const bool exhaustive_search = false;
  int returned_algo_count = 0;
  miopenConvAlgoPerf_t perfs[4];
  MIOPEN_CALL(miopenFindConvolutionForwardAlgorithm(entry_ptr->handle,
                                                    entry_ptr->conv_entry.input_desc,
                                                    x->data,
                                                    entry_ptr->conv_entry.filter_desc,
                                                    w->data,
                                                    entry_ptr->conv_entry.conv_desc,
                                                    entry_ptr->conv_entry.output_desc,
				                    y->data,
						    request_algo_count,
						    &returned_algo_count,
						    perfs,
						    entry_ptr->conv_entry.workspace,
						    workspace_size,
						    exhaustive_search
						    ));

  const std::vector<std::string> fwd_algo_names{
    "miopenConvolutionFwdAlgoGEMM",
      "miopenConvolutionFwdAlgoDirect",
      "miopenConvolutionFwdAlgoFFT",
      "miopenConvolutionFwdAlgoWinograd",
      };
  const auto best_algo = perfs[0].fwd_algo;
  LOG(INFO) << "\tMIOpen Found " << returned_algo_count << " fwd algorithms, choosing "
                         << best_algo;
  for (int i = 0; i < returned_algo_count; ++i) {
    LOG(INFO) << "\t\t" << i << ") " << fwd_algo_names[perfs[i].fwd_algo]
                         << " - time: " << perfs[i].time
                         << ", Memory: " << perfs[i].memory;
  }
  // Set Algo
  entry_ptr->conv_entry.fwd_algo = best_algo;

  const float alpha = 1.f;
  const float beta = 0.f;
  MIOPEN_CALL(miopenConvolutionForward(entry_ptr->handle,
				       &alpha,
                                       entry_ptr->conv_entry.input_desc,
                                       x->data,
                                       entry_ptr->conv_entry.filter_desc,
                                       w->data,
                                       entry_ptr->conv_entry.conv_desc,
                                       entry_ptr->conv_entry.fwd_algo,
				       &beta,
                                       entry_ptr->conv_entry.output_desc,
				       y->data,
                                       entry_ptr->conv_entry.workspace,
                                       workspace_size
                                     ));
});


TVM_REGISTER_GLOBAL("tvm.contrib.miopen.conv2d.output_shape")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  MIOpenThreadEntry* entry_ptr = MIOpenThreadEntry::ThreadLocal();
  int conv_mode = args[0];
  int pad_h = args[1];
  int pad_w = args[2];
  int stride_h = args[3];
  int stride_w = args[4];
  int dilation_h = args[5];
  int dilation_w = args[6];
  int x_dim0 = args[7];
  int x_dim1 = args[8];
  int x_dim2 = args[9];
  int x_dim3 = args[10];
  int w_dim0 = args[11];
  int w_dim1 = args[12];
  int w_dim2 = args[13];
  int w_dim3 = args[14];
  void *out_shape = args[15];
  // conv desc

  MIOPEN_CALL(miopenInitConvolutionDescriptor(entry_ptr->conv_entry.conv_desc,
					      static_cast<miopenConvolutionMode_t>(conv_mode),
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w
                                             ));
  // input desc
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.input_desc,
                                        entry_ptr->conv_entry.data_type,
                                        x_dim0,
                                        x_dim1,
                                        x_dim2,
                                        x_dim3));
  // filter desc
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.filter_desc,
                                        entry_ptr->conv_entry.data_type,
                                        w_dim0,
                                        w_dim1,
                                        w_dim2,
                                        w_dim3));

  MIOPEN_CALL(miopenGetConvolutionForwardOutputDim(entry_ptr->conv_entry.conv_desc,
                                                   entry_ptr->conv_entry.input_desc,
                                                   entry_ptr->conv_entry.filter_desc,
                                                   static_cast<int*>(out_shape),
                                                   static_cast<int*>(out_shape) + 1,
                                                   static_cast<int*>(out_shape) + 2,
                                                   static_cast<int*>(out_shape) + 3));
  });

}  // namespace contrib
}  // namespace tvm
