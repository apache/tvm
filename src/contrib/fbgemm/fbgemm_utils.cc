/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external nnpack library call.
 */
#include <fbgemm/Fbgemm.h>
#include <cstring>
#include <random>
#include <algorithm>
#include <iostream>



namespace tvm {
namespace contrib {
using namespace std;
using namespace fbgemm;

template <typename T>	
void ComputeColumnOffsets(	
    int num_rows,	
    int num_cols,	
    const T* W,	
    const vector<TensorQuantizationParams>& qparams,	
    vector<int32_t>& col_offsets) {	
  col_offsets.resize(num_cols);	
  int num_quant_groups = qparams.size();	
  for (int g = 0; g < num_quant_groups; ++g) {	
    int j_begin = g * (num_cols / num_quant_groups);	
    int j_end = j_begin + (num_cols / num_quant_groups);	
    for (int j = j_begin; j < j_end; ++j) {	
      int32_t sum = 0;	
      for (int k = 0; k < num_rows; ++k) {	
        sum += W[j * num_rows + k];	
      }	
      col_offsets[j] = sum - qparams[g].zero_point * num_rows;	
    }	
  }	
}

template void ComputeColumnOffsets<int8_t>(	
    int num_rows,	
    int num_cols,	
    const int8_t* W,	
    const vector<TensorQuantizationParams>& qparams,	
    vector<int32_t>& col_offsets);	
	
template void ComputeColumnOffsets<int16_t>(	
    int num_rows,	
    int num_cols,	
    const int16_t* W,	
    const vector<TensorQuantizationParams>& qparams,	
    vector<int32_t>& col_offsets);	

}
}
