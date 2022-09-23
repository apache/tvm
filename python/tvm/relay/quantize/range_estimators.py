# @zzk Modified from Qualcomm's Pytorch code

from calendar import c
import numpy as np
import copy
from scipy.optimize import minimize_scalar
import logging
from enum import Enum
import multiprocessing as mp
import time
import ctypes
import numba

from . import _quantize

def get_pointer(arr, ctypes_type):
    """
    Get the numpy pointer which pass to c++ end.
    """
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes_type))
    return ctypes.cast(ptr, ctypes.c_void_p)

class RangeEstimatorBase(object):
    def __init__(self, per_channel=False, quantizer=None, axis=None, n_groups=None, *args,
                 **kwargs):
        self.current_xmin = None
        self.current_xmax = None
        self.per_channel = per_channel
        self.quantizer = quantizer
        self.axis = axis
        self.n_groups = n_groups

        self.per_group_range_estimation = False
        self.ranges = None

        # Works for activation, since step 1 already decide the activation's min max
        self.max_pos_thr_out = None
        self.max_neg_thr_out = None
        self.one_sided_dist = None
        self.data_tmp = None
    
    def calibrate(self, x):
        """
        Accepts an input tensor, updates the current estimates of x_min and x_max
        and returns them.
        Parameters
        ----------
        x: Input tensor

        Returns
        -------
        self.current_xmin: tensor
        self.current_xmax: tensor
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the range estimator.
        """
        self.current_xmin = None
        self.current_xmax = None
    
    def set_min_max(self, min_val, max_val):
        self.max_pos_thr_out = max_val
        self.max_neg_thr_out = min_val
        self.one_sided_dist = bool(self.max_neg_thr_out >= 0)

class NoDataPassedError(Exception):
    """Raised data has been passed inot the Range Estimator."""

    def __init__(self):
        super().__init__('Data must be pass through the range estimator to be initialized')

    
class CurrentMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, percentile=None, *args, **kwargs):
        self.percentile = percentile
        super().__init__(*args, **kwargs)

    def calibrate(self, x):
        if self.per_group_range_estimation:
            assert self.axis !=0
            x = x.swapaxes(0, self.axis)
            x = x.reshape(x.shape[0], -1)

            ranges = np.max(x, axis=-1) - np.min(x, axis=-1)

            if self.ranges is None:
                self.ranges = ranges
            else:
                momentum = 0.1
                self.ranges = momentum * ranges + (1 - momentum) * ranges
            return
        
        if self.axis is not None:
            if self.axis != 0:
                x = x.swapaxes(0, self.axis)
            x = x.reshape(x.shape[0], -1)

            if self.n_groups is not None:
                ng = self.n_groups
                assert ng > 0 and x.shape[0] % ng == 0
                gs = x.shape[0] // ng

                # permute
                if self.ranges is not None:
                    i = np.argsort(self.ranges)
                    I = np.eye(len(i))
                    P = I[i]
                    x = np.matmul(P, x)

                x = x.reshape(ng, -1)
                m = np.min(x, axis=-1)
                M = np.max(x, axis=-1)

                m = m.repeat(m)
                M = M.repeat(M)

                # permute back
                if self.ranges is not None:
                    m = np.dot(P.T, m)
                    M = np.dot(P.T, M)
                
                self.current_xmin = m
                self.current_xmax = M
        
            else:
                self.current_xmin = x.min(axis=-1)
                self.current_xmax = x.max(axis=-1)

        elif self.per_channel:
            # Along 1st dim
            x_flattened = x.reshape(x.shape[0], -1)
            if self.percentile:
                data_np = x_flattened
                x_min, x_max = np.percentile(
                    data_np, (self.percentile, 100 - self.percentile), axis=-1
                )
                self.current_xmin = x_min
                self.current_xmax = x_max
            else:
                self.current_xmin = x_flattened.min(axis=-1)
                self.current_xmax = x_flattened.max(axis=-1)
        
        else:
            if self.percentile:
                x_min, x_max = np.percentile(x, (self.percentile, 100))
                x_min = np.atleast_1d(x_min)
                x_max = np.atleast_1d(x_max)
                self.current_xmin = x_min
                self.current_xmax = x_max
            else:
                self.current_xmin = x.min()
                self.current_xmax = x.max()
        
        return self.current_xmin, self.current_xmax

class AllMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calibrate(self, x):
        if self.per_channel:
            # Along 1st dim
            x_flattened = x.reshape(x.shape[0], -1)
            x_min = x_flattened.min(axis=-1)
            x_max = x_flattened.max(axis=-1)
        else:
            x_min = x.min()
            x_max = x.max()
        
        if self.current_xmin is None:
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            self.current_xmin = np.minimum(self.current_xmin, x_min)
            self.current_xmax = np.maximum(self.current_xmax, x_max)

        return self.current_xmin, self.current_xmax

class RunningMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, momentum=0.9, *args, **kwargs):
        self.momentum = momentum
        super().__init__(*args, **kwargs)
    
    def calibrate(self, x):
        if self.axis is not None:
            if self.axis != 0:
                x = x.swapaxes(0, self.axis)
            x = x.reshape(x.shape[0], -1)

            if self.n_groups is not None:
                ng = self.n_groups
                assert ng >0 and x.shape[0] % ng == 0
                gs = x.shape[0] // ng

                x = x.reshape(ng, -1)
                m = x.min(axis=-1)
                M = x.max(axis=-1)

                x_min = m.repeat(gs)
                x_max = M.repeat(gs)
            
            else:
                x_min = x.min(axis=-1)
                x_max = x.max(axis=-1)
            
        elif self.per_channel:
            #Along 1st dim
            x_flattened = x.reshape(x.shape[0], -1)
            x_min = x_flattened.min(axis=-1)
            x_max = x_flattened.max(axis=-1)
        
        else:
            x_min = x.min()
            x_max = x.max()

        if self.current_xmin is None:
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            self.current_xmin = (1 - self.momentum) * x_min + self.momentum * self.current_xmin
            self.current_xmax = (1 - self.momentum) * x_max + self.momentum * self.current_xmax
        
        return self.current_xmin, self.current_xmax

class OptMethod(Enum):
    grid = 1
    golden_section = 2

    @classmethod
    def list(cls):
        return [m.name for m in cls]

class MSE_Estimator(RangeEstimatorBase):
    def __init__(self, num_candidates=100, opt_method=OptMethod.grid, range_margin=0.5, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert opt_method in OptMethod

        self.opt_method = opt_method
        self.num_candidates = num_candidates
        self.loss_array = None
        self.max_pos_thr = None
        self.max_neg_thr = None
        self.max_search_range = None
        self.range_margin = range_margin

        if self.quantizer is None:
            raise NotImplementedError(
                'A Quantizer must be given as an argument to the MSE Range Estimator'
            )
        self.max_int_skew = (2 ** self.quantizer.n_bits) // 4 # for asymmetric quantization

    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False):
        y = self.quantize(data, x_min=neg_thr, x_max=pos_thr)
        temp_sum = np.sum(np.power(np.subtract(data, y), 2).reshape(len(data), -1), axis=1)
        # if we want to return the MSE loss of each channel separately, speeds up the per-channel
        # grid search
        if per_channel_loss:
            return temp_sum
        else:
            return np.sum(temp_sum)
    
    # @property
    # def step_size(self):
    #     if self.one_sided_dist is None:
    #         raise NoDataPassedError()
        
    #     return self.max_search_range / self.num_candidates
    
    # @property
    # def optimization_method(self):
    #     if self.one_sided_dist is None:
    #         raise NoDataPassedError()
        
    #     if self.opt_method == OptMethod.grid:
    #         # Grid search method
    #         if self.one_sided_dist or self.quantizer.symmetric:
    #             # 1-D grid search
    #             return self._perform_1D_search
    #         else:
    #             # 2-D grid search
    #             return self._perform_2D_search
    #     elif self.opt_method == OptMethod.golden_section:
    #         # Golden section method
    #         if self.one_sided_dist or self.quantizer.symmetric:
    #             return self._golden_section_symmetric
    #         else:
    #             return self._golden_section_asymmetric
    #     else:
    #         raise NotImplementedError('Optimization Method not Implemented')

    @property
    def step_size(self):
        if self.one_sided_dist is None:
            return -1
        
        return self.max_search_range / self.num_candidates
    
    def optimization_method(self):
        if self.one_sided_dist is None:
            raise NoDataPassedError()
        
        if self.opt_method == OptMethod.grid:
            # Grid search method
            if self.one_sided_dist or self.quantizer.symmetric:
                # 1-D grid search
                return self._perform_1D_search
            else:
                # 2-D grid search
                return self._perform_2D_search
        elif self.opt_method == OptMethod.golden_section:
            # Golden section method
            if self.one_sided_dist or self.quantizer.symmetric:
                return self._golden_section_symmetric
            else:
                return self._golden_section_asymmetric
        else:
            raise NotImplementedError('Optimization Method not Implemented')
    

    def quantize(self, x_float, x_min=None, x_max=None):
        # zzk_debug: original code set per_channel to be False here, why?
        # maybe because in transfomer there is no need to use per-channel format?
        # answer: because in quantization stage, it is just searching, no need to perform per-channel 
        # quantization. 
        # In the current implementation no optimization procesure requires temp quantizer for
        # loss fx to be per-channel
        temp_q = copy.deepcopy(self.quantizer)
        temp_q.per_channel = False
        if x_min or x_max:
            temp_q.set_quant_range(x_min, x_max)
        return temp_q(x_float)
    
    def golden_sym_loss(self, range, data):
        """
        Loss function passed to the golden section optimizer from scipy in case of symmetric
        quantization
        """
        neg_thr = 0 if self.one_sided_dist else -range
        pos_thr = range
        return self.loss_fx(data, neg_thr, pos_thr)
    
    def golden_asym_shift_loss(self, shift, range, data):
        """
        Inner Loss function (shift) passed to the golden section optimizer from scipy
        in case of asymmetric quantization
        """
        pos_thr = range + shift
        neg_thr = -range + shift
        return self.loss_fx(data, neg_thr, pos_thr)
    
    def golden_asym_range_loss(self, range, data):
        """
        Outer Loss function (range) passed to the golden section optimizer from scipy in case of
        asymmetric quantization
        """
        temp_delta = 2 * range / (2 ** self.quantizer.n_bits - 1)
        max_shift = temp_delta * self.max_int_skew
        result = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(range, data),
                bounds=(-max_shift, max_shift),
                method='Bounded',
        )
        return result.fun

    def _define_search_range(self, data):
        self.channel_groups = len(data) if self.per_channel else 1
        self.current_xmax = np.zeros(self.channel_groups)
        self.current_xmin = np.zeros(self.channel_groups)

        if self.one_sided_dist or self.quantizer.symmetric:
            # 1D search space
            self.loss_array = np.zeros(
                (self.channel_groups, self.num_candidates + 1)
            ) # 1D search space
            self.loss_array[:, 0] = np.inf # exclude interval_start=interval_finish
            # Defining the search range for clopping thresholds
            if self.max_pos_thr_out is None:
                self.max_pos_thr = max(abs(float(data.min())), float(data.max())) + self.range_margin
                self.max_neg_thr = -self.max_pos_thr
            else:
                self.max_pos_thr = max(abs(self.max_neg_thr_out), self.max_pos_thr_out) + self.range_margin
                self.max_neg_thr = -self.max_pos_thr
            
            self.max_search_range = self.max_pos_thr
        else:
            # 2D search space (3rd and 4th index correspond to asymmetry where fourth
            # index represents whether the skew is positive (0) or negative (1))
            self.loss_array = np.zeros(
                [self.channel_groups, self.num_candidates + 1, self.max_int_skew, 2]
            ) # 2D search space
            self.loss_array[:, 0, :, :] = np.inf # exclude interval_start=interval_finish
            # Define the search range for clipping thresholds in asymmetric case
            if self.max_pos_thr_out is None:
                self.max_pos_thr = float(data.max()) + self.range_margin
                self.max_neg_thr = float(data.min()) - self.range_margin
            else:
                self.max_pos_thr = self.max_pos_thr_out + self.range_margin
                self.max_neg_thr = self.max_neg_thr_out - self.range_margin
            
            self.max_search_range = max(abs(self.max_pos_thr), abs(self.max_neg_thr))
    
    def _perform_1D_search(self, data):
        """
        Grid search through all candidate quantizers in 1D to find the best
        The loss is accmulated over all batches without any momentum
        :param data: input tensor
        """
        self.data_tmp = data

        for cand_index in range(1, self.num_candidates + 1):
            neg_thr = 0 if self.one_sided_dist else -self.step_size * cand_index
            pos_thr = self.step_size * cand_index

            self.loss_array[:, cand_index] += self.loss_fx(
                self.data_tmp, neg_thr, pos_thr, per_channel_loss=self.per_channel
            )

        # find the best clipping thresholds
        min_cand = self.loss_array.argmin(axis=1)
        xmin = (
            np.zeros(self.channel_groups) if self.one_sided_dist else -self.step_size * min_cand
        ).astype(np.single)
        xmax = (self.step_size * min_cand).astype(np.single)
        self.current_xmax = xmax
        self.current_xmin = xmin
    
    def _perform_2D_search(self, data):
        """
        Grid search through all candidate quantizers in 2D to find the best
        The loss is accumulated over all batches withou any momentum
        Parameters
        ----------
        data : Numpy Tensor
        Returns
        ----------

        """
        self.data_tmp = data

        for cand_index in range(1, self.num_candidates + 1):
            temp_start = -self.step_size * cand_index
            temp_finish = self.step_size * cand_index
            temp_delta = float(temp_finish - temp_start) / (2 ** self.quantizer.n_bits - 1)
            for shift in range(self.max_int_skew):
                for reverse in range(2):
                    # introducint asymmetry in the quantization range
                    skew = ((-1) ** reverse) * shift * temp_delta
                    neg_thr = max(temp_start + skew, self.max_neg_thr)
                    pos_thr = min(temp_finish + skew, self.max_pos_thr)

                    self.loss_array[:, cand_index, shift, reverse] += self.loss_fx(
                        self.data_tmp, neg_thr, pos_thr, per_channel_loss=self.per_channel
                    )
        
        for channel_index in range(self.channel_groups):
            min_cand, min_shift, min_reverse = np.unravel_index(
                np.argmin(self.loss_array[channel_index], axis=None),
                self.loss_array[channel_index].shape,
            )
            min_interval_start = -self.step_size * min_cand
            min_interval_finish = self.step_size * min_cand
            min_delta = float(min_interval_finish - min_interval_start) / (
                2 ** self.quantizer.n_bits - 1
            )
            min_skew = ((-1) ** min_reverse) * min_shift * min_delta
            xmin = max(min_interval_start + min_skew, self.max_neg_thr)
            xmax = min(min_interval_finish + min_skew, self.max_pos_thr)

            self.current_xmin[channel_index] = xmin
            self.current_xmax[channel_index] = xmax

    def _golden_section_symmetric(self, data):
        for channel_index in range(self.channel_groups):
            if channel_index == 0 and not self.per_channel:
                data_segment = data
            else:
                data_segment = data[channel_index]
            
            self.result = minimize_scalar(
                self.golden_asym_range_loss,
                args=data_segment,
                bounds=(0.01 * self.max_search_range, self.max_search_range),
                method='Bounded',
            )
            self.current_xmax[channel_index] = self.result.x
            self.current_xmin[channel_index] = np.zeros_like(self.current_xmax[channel_index]) if self.one_sided_dist \
                                                else -self.current_xmax[channel_index]
    
    def _golden_section_asymmetric(self, data):
        for channel_index in range(self.channel_groups):
            if channel_index == 0 and not self.per_channel:
                data_segment = data
            else:
                data_segment = data[channel_index]
            
            self.result = minimize_scalar(
                self.golden_asym_range_loss,
                args=data_segment,
                bounds=(0.01 * self.max_search_range, self.max_search_range),
                method='Bounded',
            )
            self.final_range = self.result.x
            temp_delta = 2 * self.final_range / (2 ** self.quantizer.n_bits - 1)
            max_shift = temp_delta * self.max_int_skew
            self.subresult = minimize_scalar(
                self.golden_asym_shift_loss,

            )
            self.current_xmax[channel_index] = self.final_range + self.final_shift
            self.current_xmin[channel_index] = -self.final_range + self.final_shift
    
    def calibrate(self, data):
        if self.loss_array is None:
            # Initialize search range on first batch, and accumulate losses with subsequent calls
            # Decide whether input distribution is one-sided
            if self.one_sided_dist is None:
                self.one_sided_dist = bool((data.min() >= 0))
            
            # Define search
            self._define_search_range(data)

        # Perform Search/Optimization for Quantization Ranges
        self.optimization_method()(data)

        return self.current_xmin, self.current_xmax

    def reset(self):
        super().reset()
        self.loss_array = None

def log_softmax(x, axis=None):
    x -= np.max(x, axis = 1, keepdims = True)
    x_softmax = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    x_log_softmax = np.log(x_softmax)
    return x_log_softmax

def softmax(x, axis=None):
    x -= np.max(x, axis = 1, keepdims = True)
    x_softmax = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x_softmax

#zzk_debug: original versions only works for activation?
class CrossEntropyEstimator(MSE_Estimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # per channel loss argument is here only to be consistent in definition with other loss fxs
    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False):
        quantized_data = self.quantize(data, neg_thr, pos_thr)
        log_quantized_probs = log_softmax(quantized_data, axis=1)
        unquantized_probs = softmax(data, axis=1)
        if per_channel_loss:
            return -unquantized_probs * log_quantized_probs
        else:
            return np.sum(-unquantized_probs * log_quantized_probs)

class CosineSimilarityEstimator(MSE_Estimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False):
        quantized_data = self.quantize(data, neg_thr, pos_thr)
        if per_channel_loss:
            quantized_data_tmp = quantized_data.reshape((len(quantized_data), -1))
            unquantized_data_tmp = data.reshape((len(data), -1))
            cosine_per_channel = []
            for i in range(len(data)):
                num = float(np.dot(quantized_data_tmp[i], unquantized_data_tmp[i]))
                denom = np.linalg.norm(quantized_data_tmp[i]) * np.linalg.norm(unquantized_data_tmp)
                result_tmp = 0.5 + 0.5 * (num / denom) if denom != 0 else 0
                cosine_per_channel.append(result_tmp)
            cosine_per_channel = np.array(cosine_per_channel)
            return cosine_per_channel
        else:
            quantized_data_tmp = quantized_data.flatten()
            unquantized_data_tmp = data.flatten()
            num = float(np.dot(quantized_data_tmp, unquantized_data_tmp))
            denom = np.linalg.norm(quantized_data_tmp) * np.linalg.norm(unquantized_data_tmp)
            return 0.5 + 0.5 * (num / denom) if denom !=0 else 0


# class KLDivergence(RangeEstimatorBase):
#     """ 
#         Calculate KL Divergence, since tvm's version doesn't support
#         asymmetric quantization, we reimplement it here.
#     """
#     def __init__(self, num_bins=8001, range_margin=0.5, *args,
#                  **kwargs):
#         super().__init__(*args, **kwargs)

#         self.num_bins = num_bins
#         self.loss_array = None
#         self.max_pos_thr = None
#         self.max_neg_thr = None
#         self.max_search_range = None
#         self.range_margin = range_margin
#         if self.quantizer is None:
#             raise NotImplementedError(
#                 'A Quantizer must be given as an argument to the MSE Range Estimator'
#             )
#         self.num_quantized_bins = 2 ** self.quantizer.n_bits - 1
#         self.num_candidate_bins = self.num_bins // 2 + 1 - self.num_quantized_bins // 2
    
#     @property
#     def optimization_method(self):
#         if self.one_sided_dist is None:
#             raise NoDataPassedError()
        

#         if self.one_sided_dist or self.quantizer.symmetric:
#             # 1-D grid search
#             return self._perform_1D_search
#         else:
#             # 2-D grid search
#             #return self._perform_2D_search
#             return self._perform_1D_search

#     def loss_fx(self, p, q, per_channel_loss=False):
#         # loss array [channel groups, num bins] for symmetric quantization
#         # loss array [channel groups, num bins, max_int_skew, 2] for asymmetric quantization
#         # p or q format is [channel groups, bins]
#         if per_channel_loss:
#             p_sum = np.sum(p, axis=1)
#             q_sum = np.sum(q, axis=1)
#         else:
#             p_sum = np.sum(p)
#             q_sum = np.sum(q)
        
#         if per_channel_loss:
#             for channel_index in range(self.channel_groups):
#                 kl_loss = np.zeros((p.shape[0],))
#                 p[channel_index] = p[channel_index] / p_sum[channel_index]
#                 q[channel_index] = q[channel_index] / q_sum[channel_index]
#                 div_tmp = np.divide(p[channel_index], q[channel_index], 
#                                     out=np.zeros_like(p[channel_index]), where=q[channel_index]!=0)
#                 log_div_tmp = np.log(div_tmp, out=np.zeros_like(div_tmp), where=div_tmp!=0)
#                 kl_loss[channel_index] =  np.sum(p[channel_index] * log_div_tmp)
#         else:
#             p = p / p_sum
#             q = q / q_sum
#             div_tmp = np.divide(p, q, out=np.zeros_like(p), where=q!=0)
#             log_div_tmp = np.log(div_tmp, out=np.zeros_like(div_tmp), where=div_tmp!=0)
#             kl_loss = np.sum(p * log_div_tmp)
        
#         return kl_loss
    
#     def _define_search_range(self, data):
#         self.channel_groups = len(data) if self.per_channel else 1
#         self.current_xmax = np.zeros(self.channel_groups)
#         self.current_xmin = np.zeros(self.channel_groups)

#         if self.one_sided_dist or self.quantizer.symmetric:
#             # 1D search space
#             self.loss_array = np.zeros(
#                 (self.channel_groups, self.num_candidate_bins + 1)
#             ) # 1D search space
#             self.loss_array[:, 0] = np.inf # exclude interval_start=interval_finish
#             # Defining the search range for clopping thresholds
#             if self.max_pos_thr_out is None:
#                 self.max_pos_thr = max(abs(float(data.min())), float(data.max()))
#                 self.max_neg_thr = -self.max_pos_thr
#             else:
#                 self.max_pos_thr = max(abs(self.max_neg_thr_out), self.max_pos_thr_out)
#                 self.max_neg_thr = -self.max_pos_thr
            
#             self.max_search_range = self.max_pos_thr
#         else:
#             # # 2D search space (3rd and 4th index correspond to asymmetry where fourth
#             # # index represents whether the skew is positive (0) or negative (1))
#             # self.loss_array = np.zeros(
#             #     [self.channel_groups, self.num_candidate_bins + 1, self.max_int_skew, 2]
#             # ) # 2D search space
#             # self.loss_array[:, 0, :, :] = np.inf # exclude interval_start=interval_finish
#             # # Define the search range for clipping thresholds in asymmetric case
#             # if self.max_pos_thr_out is None:
#             #     self.max_pos_thr = float(data.max()) + self.range_margin
#             #     self.max_neg_thr = float(data.min()) - self.range_margin
#             # else:
#             #     self.max_pos_thr = self.max_pos_thr_out + self.range_margin
#             #     self.max_neg_thr = self.max_neg_thr_out - self.range_margin
        
#             # self.max_search_range = max(abs(self.max_pos_thr), abs(self.max_neg_thr))

#             # current solution

#             # 1D search space
#             self.loss_array = np.zeros(
#                 (self.channel_groups, self.num_candidate_bins + 1)
#             ) # 1D search space
#             self.loss_array[:, 0] = np.inf # exclude interval_start=interval_finish
#             # Defining the search range for clopping thresholds
#             if self.max_pos_thr_out is None:
#                 self.max_pos_thr = max(abs(float(data.min())), float(data.max()))
#                 self.max_neg_thr = -self.max_pos_thr
#             else:
#                 self.max_pos_thr = max(abs(self.max_neg_thr_out), self.max_pos_thr_out)
#                 self.max_neg_thr = -self.max_pos_thr
            
#             self.max_search_range = self.max_pos_thr

#     def smoothDistribution(self, p, eps = 0.0001):
#         p_copy = p.copy()
#         p_zero_copy = p.copy()
#         p_nonzero_copy = p.copy()
#         p_zero_copy[p_zero_copy != 0] = 0
#         p_zero_copy[p_zero_copy == 0] = 1
#         p_nonzero_copy[p_nonzero_copy == 0] = 0
#         p_nonzero_copy[p_nonzero_copy != 0] = 1
#         if self.per_channel:
#             n_zeros = np.sum(p_zero_copy, axis=1)
#             n_nonzeros = np.sum(p_nonzero_copy, axis=1)
#             assert np.any(n_nonzeros)
#             eps1 = eps * n_zeros / n_nonzeros
#             assert eps1 < 1
#             p_copy += eps * p_zero_copy - eps1 * p_nonzero_copy
#         else:
#             n_zeros = np.sum(p_zero_copy)
#             n_nonzeros = np.sum(p_nonzero_copy)
#             assert np.any(n_nonzeros)
#             eps1 = eps * n_zeros / n_nonzeros
#             assert eps1 < 1
#             p_copy += eps * p_zero_copy - eps1 * p_nonzero_copy

#         return p_copy
    
#     def _perform_1D_search(self, data):
#         """
#         Grid search through all candidate quantizers in 1D to find the best
#         The loss is accmulated over all batches without any momentum
#         :param data: input tensor
#         """
#         if self.per_channel:
#             hist = []
#             hist_edges = []
#             quantized_bins = np.zeros((self.channel_groups, self.num_quantized_bins))
#             for channel_index in range(self.channel_groups):
#                 hist_tmp, hist_edges_tmp = np.histogram(data[channel_index], bins=self.num_bins,
#                                                     range=(self.max_neg_thr, self.max_pos_thr))
#                 hist.append(hist_tmp)
#                 hist_edges.append(hist_edges_tmp)
#                 hist = np.array(hist)
#                 hist_edges = np.array(hist_edges)

#             for cand_index in range(1, self.num_candidate_bins + 1):
#                 p_bin_idx_start = self.num_bins // 2 - self.num_quantized_bins // 2 - cand_index + 1
#                 p_bin_idx_stop = self.num_bins // 2 + self.num_quantized_bins // 2 + cand_index

#                 p_tmp = np.zeros((len(hist), p_bin_idx_stop - p_bin_idx_start))
#                 sliced_nd_hist = np.zeros_like(p_tmp)

#                 for j in range(self.num_bins):
#                     if (j <= p_bin_idx_start):
#                         p_tmp[:,0] += hist[:,j]
#                     elif (j >= p_bin_idx_stop):
#                         p_tmp[:,-1] += hist[:,j]
#                     else:
#                         sliced_nd_hist[:, j-p_bin_idx_start] = hist[:, j]
#                         p_tmp[:, j-p_bin_idx_start] = hist[:, j]
                
#                 num_merged_bins = sliced_nd_hist.shape[1] // self.num_quantized_bins
#                 for j in range(self.num_quantized_bins):
#                     start = j * num_merged_bins
#                     stop = (j + 1) * num_merged_bins
#                     quantized_bins[:, j] = np.sum(sliced_nd_hist[:, start:stop], axis = 1)
#                 # deal with the tile if it is existed
#                 quantized_bins[:, -1] += np.sum(sliced_nd_hist[:, self.num_quantized_bins * num_merged_bins:], axis = 1)

#                 # expand quantized bins into p.size bins
#                 q_tmp = np.zeros_like(sliced_nd_hist)
#                 for j in range(self.num_quantized_bins):
#                     start = j * num_merged_bins
#                     stop = q_tmp.shape[1] if j == self.num_quantized_bins - 1 else (j+1) * num_merged_bins
#                     norm = np.sum(sliced_nd_hist[:,start:stop]!=0, axis = 1)
#                     for k in range(start, stop):
#                         q_tmp[:, k] = np.divide(quantized_bins[:, j], norm, 
#                                 out=np.zeros_like(quantized_bins[:, j]), where=p_tmp[:, k]!=0 and norm[k]!=0)
                
#                 p_tmp = self.smoothDistribution(p_tmp)
#                 q_tmp = self.smoothDistribution(q_tmp)

#                 self.loss_array[:, cand_index] += self.loss_fx(
#                     p_tmp, q_tmp, per_channel_loss=self.per_channel
#                 )
#             min_cand = self.loss_array.argmin(axis=1)
#             x_min_kl = hist_edges[:, self.num_bins // 2 + self.num_quantized_bins // 2 + min_cand]
#             xmin = (
#                 np.zeros(self.channel_groups) if self.one_sided_dist else -x_min_kl
#             ).astype(np.single)
#             xmax = x_min_kl.astype(np.single)
#             # self.current_xmax = xmax
#             # self.current_xmin = xmin

#             #elementwise compare
#             self.current_xmax = np.minimum(xmax, self.max_pos_thr)
#             self.current_xmin = np.minimum(np.maximum(xmin, self.max_neg_thr), 0)
#             if(self.quantizer.symmetric):
#                 assert((np.abs(self.current_xmin) == np.abs(self.current_xmax)).all())
            
#         else:
#             quantized_bins = np.zeros((self.num_quantized_bins, ))
#             hist_tmp, hist_edges_tmp = np.histogram(data, bins=self.num_bins,
#                                                 range=(self.max_neg_thr, self.max_pos_thr))
#             hist = hist_tmp
#             hist_edges = hist_edges_tmp

#             for cand_index in range(1, self.num_candidate_bins + 1):
#                 p_bin_idx_start = self.num_bins // 2 - self.num_quantized_bins // 2 - cand_index + 1
#                 p_bin_idx_stop = self.num_bins // 2 + self.num_quantized_bins // 2 + cand_index                                    
                
#                 # now we get the referenced p and q, smooth them.
#                 p_tmp = np.zeros((p_bin_idx_stop - p_bin_idx_start, ))
#                 sliced_nd_hist = np.zeros_like(p_tmp)

#                 for j in range(self.num_bins):
#                     if (j <= p_bin_idx_start):
#                         p_tmp[0] += hist[j]
#                     elif (j >= p_bin_idx_stop):
#                         p_tmp[-1] += hist[j]
#                     else:
#                         sliced_nd_hist[j - p_bin_idx_start] = hist[j]
#                         p_tmp[j - p_bin_idx_start] = hist[j]
                
#                 num_merged_bins = sliced_nd_hist.shape[0] // self.num_quantized_bins
#                 for j in range(self.num_quantized_bins):
#                     start = j * num_merged_bins
#                     stop = (j + 1) * num_merged_bins
#                     quantized_bins[j] = np.sum(sliced_nd_hist[start:stop])
#                 # deal with the tile if it is existed
#                 quantized_bins[-1] += np.sum(sliced_nd_hist[self.num_quantized_bins * num_merged_bins:])

#                 # expand quantized bins into p.size bins
#                 q_tmp = np.zeros_like(sliced_nd_hist)
        
#                 for j in range(self.num_quantized_bins):
#                     start = j * num_merged_bins
#                     stop = q_tmp.shape[0] if j == self.num_quantized_bins - 1 else (j + 1) * num_merged_bins
#                     norm = np.sum(sliced_nd_hist[start:stop]!=0)
#                     if(norm != 0):
#                         for k in range(start, stop):
#                             q_tmp[k] = np.divide(quantized_bins[j], norm, out=np.zeros_like(quantized_bins[j]), where=p_tmp[k]!=0)

#                 p_tmp = self.smoothDistribution(p_tmp)
#                 q_tmp = self.smoothDistribution(q_tmp)

#                 self.loss_array[:, cand_index] += self.loss_fx(
#                     p_tmp, q_tmp, per_channel_loss=self.per_channel
#                 )

#             min_cand = self.loss_array.argmin(axis=1)
#             x_min_kl = hist_edges[self.num_bins // 2 + self.num_quantized_bins // 2 + min_cand]
#             xmin = (
#                 np.zeros(self.channel_groups) if self.one_sided_dist else -x_min_kl
#             ).astype(np.single)
#             xmax = x_min_kl.astype(np.single)
#             # self.current_xmax = xmax
#             # self.current_xmin = xmin

#             #elementwise compare
#             self.current_xmax = np.minimum(xmax, self.max_pos_thr)
#             self.current_xmin = np.minimum(np.maximum(xmin, self.max_neg_thr), 0)
#             if(self.quantizer.symmetric):
#                 assert((np.abs(self.current_xmin) == np.abs(self.current_xmax)).all())
    
#     def _perform_2D_search(self, data):
#         """
#         Grid search through all candidate quantizers in 2D to find the best
#         The loss is accumulated over all batches withou any momentum
#         Parameters
#         ----------
#         data : Numpy Tensor
#         Returns
#         ----------
#         """
#         # zzk_debug: haven't decide how to implement kl-divergence in asymmetic yet
#         raise NotImplementedError()
    
#     def calibrate(self, data):
#         if self.loss_array is None:
#             if self.one_sided_dist is None:
#                 self.one_sided_dist = bool((data.min() >= 0))
            
#             # Define search
#             self._define_search_range(data)
        
#         self.optimization_method(data)

#         return self.current_xmin, self.current_xmax


def kl_divergence_intrin(data_tmp):
    hist_tmp = data_tmp[0]
    hist_edges_tmp = data_tmp[1]
    num_bins = data_tmp[2]
    num_quantized_bins = data_tmp[3]
    num_candidate_bins = data_tmp[4]
    hist_tmp_ptr = get_pointer(hist_tmp.astype(np.int32), ctypes.c_int)
    hist_edges_tmp_ptr = get_pointer(hist_edges_tmp.astype(np.float32), ctypes.c_float)
    divergence_tmp = np.zeros((num_candidate_bins,)).astype(np.float32)
    divergence_tmp_ptr = get_pointer(divergence_tmp, ctypes.c_float)
    _quantize.FindScaleByKLMinimization(hist_tmp_ptr, hist_edges_tmp_ptr, divergence_tmp_ptr, num_bins, num_quantized_bins)
    return divergence_tmp


class KLDivergence(RangeEstimatorBase):
    """ 
        Calculate KL Divergence, since tvm's version doesn't support
        asymmetric quantization, we reimplement it here.
    """
    def __init__(self, num_bins=8001, range_margin=0.5, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.num_bins = num_bins
        self.loss_array = None
        self.max_pos_thr = None
        self.max_neg_thr = None
        self.max_pos_thr_bin = None
        self.max_neg_thr_bin = None
        self.hist = None
        self.range_margin = range_margin
        if self.quantizer is None:
            raise NotImplementedError(
                'A Quantizer must be given as an argument to the MSE Range Estimator'
            )
        self.num_quantized_bins = 2 ** self.quantizer.n_bits - 1
        self.num_candidate_bins = self.num_bins // 2 + 1 - self.num_quantized_bins // 2
    
    # @property
    # def optimization_method(self):
    #     if self.one_sided_dist is None:
    #         raise NoDataPassedError
    
    #     return self._perform_1D_search

    def optimization_method(self):
        if self.one_sided_dist is None:
            raise NoDataPassedError

        return self._perform_1D_search

    def _define_search_range(self, data):
        self.channel_groups = len(data) if self.per_channel else 1
        self.current_xmax = np.zeros(self.channel_groups)
        self.current_xmin = np.zeros(self.channel_groups)

        # 1D search space
        self.loss_array = np.zeros(
            (self.channel_groups, self.num_candidate_bins)
        ) # 1D search space

        if self.max_pos_thr_out is None:
            self.max_pos_thr_bin = max(abs(float(data.min())), float(data.max()))
            self.max_neg_thr_bin = -self.max_pos_thr_bin
        else:
            self.max_pos_thr_bin = max(abs(self.max_neg_thr_out), self.max_pos_thr_out)
            self.max_neg_thr_bin = -self.max_pos_thr_bin

        if self.one_sided_dist or self.quantizer.symmetric:
            # Defining the search range for clopping thresholds
            if self.max_pos_thr_out is None:
                self.max_pos_thr = max(abs(float(data.min())), float(data.max()))
                self.max_neg_thr = -self.max_pos_thr
            else:
                self.max_pos_thr = max(abs(self.max_neg_thr_out), self.max_pos_thr_out)
                self.max_neg_thr = -self.max_pos_thr
        else:
            if self.max_pos_thr_out is None:
                self.max_pos_thr = float(data.max())
                self.max_neg_thr = float(data.min())
            else:
                self.max_pos_thr = self.max_pos_thr_out
                self.max_neg_thr = self.max_neg_thr_out

    
    def _perform_1D_search(self, data):
        """
        Grid search through all candidate quantizers in 1D to find the best
        The loss is accmulated over all batches without any momentum
        :param data: input tensor
        """
        if self.per_channel:
            hist_edges = []
            hist = []
            pc_max_neg_thr_bin = []
            pc_max_pos_thr_bin = []
    
            for channel_index in range(self.channel_groups):
                tmp_max_pos_thr = max(abs(float(data[channel_index].min())), float(data[channel_index].max()))
                tmp_max_neg_thr = -tmp_max_pos_thr
                pc_max_pos_thr_bin.append(tmp_max_pos_thr)
                pc_max_neg_thr_bin.append(tmp_max_neg_thr)
                hist_tmp, hist_edges_tmp = np.histogram(data[channel_index], bins=self.num_bins,
                                    range=(self.max_neg_thr_bin, self.max_pos_thr_bin))
                hist_edges.append(hist_edges_tmp)
                hist.append(hist_tmp)
            
            hist_edges = np.array(hist_edges)
            hist = np.array(hist)
            pc_max_neg_thr_bin = np.array(pc_max_neg_thr_bin)
            pc_max_pos_thr_bin = np.array(pc_max_pos_thr_bin)

            divergence = []
            samples = [[hist[i], hist_edges[i], self.num_bins, self.num_quantized_bins, self.num_candidate_bins] for i in range(self.channel_groups)]
            with mp.Pool(16) as pool:
                divergence += list(pool.map(kl_divergence_intrin, samples))

            assert(len(divergence) == self.channel_groups)
            
            for channel_index in range(self.channel_groups):
                self.loss_array[channel_index, :] = divergence[channel_index]

            # old code
            # for channel_index in range(self.channel_groups): 
            #     hist_tmp, hist_edges_tmp = np.histogram(data[channel_index], bins=self.num_bins,
            #                                         range=(self.max_neg_thr_bin, self.max_pos_thr_bin))
            #     hist_edges.append(hist_edges_tmp)
            #     hist_tmp_ptr = get_pointer(hist_tmp.astype(np.int32), ctypes.c_int)
            #     hist_edges_tmp_ptr = get_pointer(hist_edges_tmp.astype(np.float32), ctypes.c_float)
            #     divergence_tmp = np.zeros((self.num_candidate_bins,)).astype(np.float32)
            #     divergence_tmp_ptr = get_pointer(divergence_tmp, ctypes.c_float)
            #     _quantize.FindScaleByKLMinimization(hist_tmp_ptr, hist_edges_tmp_ptr, divergence_tmp_ptr, self.num_bins, self.num_quantized_bins)
            #     self.loss_array[channel_index, :] = self.loss_array[channel_index, :] + divergence_tmp
            

            min_cand = self.num_bins // 2 + self.num_quantized_bins // 2 + self.loss_array.argmin(axis=1)
            x_min_kl = hist_edges[np.arange(self.channel_groups), min_cand]
            xmin = (
                np.zeros(self.channel_groups) if self.one_sided_dist else -x_min_kl
            ).astype(np.single)
            xmax = x_min_kl.astype(np.single)

            #elementwise compare
            if(self.quantizer.symmetric):
                self.current_xmax = xmax
                self.current_xmin = xmin
            else:
                self.current_xmax = np.minimum(xmax, pc_max_pos_thr_bin)
                self.current_xmin = np.minimum(np.maximum(xmin, pc_max_neg_thr_bin), 0)
            
        else:
            hist_tmp, hist_edges_tmp = np.histogram(data, bins=self.num_bins,
                                                range=(self.max_neg_thr_bin, self.max_pos_thr_bin))

            if self.hist is None:
                self.hist = hist_tmp
            else:
                self.hist = self.hist + hist_tmp

            hist_tmp_ptr = get_pointer(self.hist.astype(np.int32), ctypes.c_int)
            hist_edges_tmp_ptr = get_pointer(hist_edges_tmp.astype(np.float32), ctypes.c_float)
            divergence_tmp = np.zeros((self.num_candidate_bins,)).astype(np.float32)
            divergence_tmp_ptr = get_pointer(divergence_tmp, ctypes.c_float)
            _quantize.FindScaleByKLMinimization(hist_tmp_ptr, hist_edges_tmp_ptr, divergence_tmp_ptr, self.num_bins, self.num_quantized_bins)
            self.loss_array[0, :] = divergence_tmp
            min_cand = self.num_bins // 2 + self.num_quantized_bins // 2 + self.loss_array.argmin(axis=1)
            x_min_kl = hist_edges_tmp[min_cand]
            xmin = (
                np.zeros(self.channel_groups) if self.one_sided_dist else -x_min_kl
            ).astype(np.single)
            xmax = x_min_kl.astype(np.single)
            # self.current_xmax = xmax
            # self.current_xmin = xmin

            #elementwise compare
            if(self.quantizer.symmetric):
                self.current_xmax = xmax
                self.current_xmin = xmin
            else:
                self.current_xmax = np.minimum(xmax, self.max_pos_thr)
                self.current_xmin = np.minimum(np.maximum(xmin, self.max_neg_thr), 0)
    
    def _perform_2D_search(self, data):
        """
        Grid search through all candidate quantizers in 2D to find the best
        The loss is accumulated over all batches withou any momentum
        Parameters
        ----------
        data : Numpy Tensor
        Returns
        ----------
        """
        # zzk_debug: haven't decide how to implement kl-divergence in asymmetric yet
        raise NotImplementedError()
    
    def calibrate(self, data):
        if self.loss_array is None:
            if self.one_sided_dist is None:
                self.one_sided_dist = bool((data.min() >= 0))
            
            # Define search
            self._define_search_range(data)
        
        self.optimization_method()(data)

        return self.current_xmin, self.current_xmax



                


            
        

        
