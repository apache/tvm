# @zzk Modified from Qualcomm's Pytorch code
# Todo list: 1. round measure.

import numpy as np

def clamp(x, min_val=-np.inf, max_val=np.inf):
    """Clamp x into [min_val, max_val]"""
    return np.clip(x, min_val, max_val)

class QuantizerNotInitializedError(Exception):
    """Raised when a quantizer has not initialized"""

    def __init__(self):
        super(QuantizerNotInitializedError, self).__init__('Quantizer has not been initialized yet')

class QuantizerBase(object):
    def __init__(self, n_bits, per_channel=False, axis=None):
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.axis = axis
    
    @property
    def is_initialized(self):
        raise NotImplementedError()

    @property
    def x_max(self):
        raise NotImplementedError()

    @property
    def symmetric(self):
        raise NotImplementedError()

    @property
    def x_min(self):
        raise NotImplementedError()

    def forward(self, x_float):
        raise NotImplementedError()

    def _adjust_params_per_axis(self, x):
        raise NotImplementedError()

    def _adjust_params_per_channel(self, x):
        raise NotImplementedError()

    def set_quant_range(self, x_min, x_max):
        raise NotImplementedError()

    def reset(self):
        self._delta = None

class AsymmetricUniformQuantizer(QuantizerBase):
    """
    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    per_channel: bool
        If True: allows for per-channel quantization
    """
    def __init__(self, name, n_bits, per_channel=False, axis=None, eps=1e-8):

        super().__init__(n_bits, per_channel)
        self._name = name
        self._delta = None
        self._zero_float = None
        self.per_channel = per_channel
        self.n_bits = n_bits
        self.axis = axis
        self.eps = eps
        self.float_max = None
        self.float_min = None
    
    @property
    def name(self):
        return self._name

    @property
    def bitwidth(self):
        return self.n_bits
    
    @property
    def perchannel(self):
        return self.per_channel
    
    @property
    def delta(self):
        if self._delta is not None:
            return self._delta
        else:
            raise QuantizerNotInitializedError()

    @property
    def zero_float(self):
        if self._zero_float is not None:
            return self._zero_float
        else:
            raise QuantizerNotInitializedError()

    @property
    def is_initialized(self):
        return self._delta is not None

    @property
    def symmetric(self):
        return False

    @property
    def int_min(self):
        # integer grid minimum
        return 0.0

    @property
    def int_max(self):
        # integer grid maximum
        return 2.0 ** self.n_bits - 1

    @property
    def scale(self):
        return clamp(self.delta, min_val=self.eps)
    
    @property
    def zero_point(self):
        zero_point = np.round(self.zero_float)
        zero_point = clamp(zero_point, self.int_min, self.int_max)
        return zero_point

    @property
    def x_max(self):
        return self.scale * (self.int_max - self.zero_point)

    @property
    def x_min(self):
        return self.scale * (self.int_min - self.zero_point)

    def to_integer_forward(self, x_float):
        """
        Qunatized input to its integer represantion
        Parameters
        ----------
        x_float: Numpy Float Tensor
                Full-precision Tensor

        Returns
        -------
        x_int: Numpy Float Tensor of integers
        """
        x_int = np.round(x_float / self.scale) + self.zero_point
        x_int = np.clip(x_int, self.int_min, self.int_max)

        return x_int
    
    def __call__(self, x_float):
        """
        Quantizes (quantized to integer and the scales back to original domain)
        Parameters
        ----------
        x_float: Numpy Float Tensor
            Full-precision Tensor

        Returns
        -------
        x_quant: Numpy Float Tensor
            Quantized-Dequantized Tensor
        """
        if self.axis is not None:
            self._adjust_params_per_axis(x_float)
        
        if self.per_channel:
            self._adjust_params_per_channel(x_float)

        x_int = self.to_integer_forward(x_float)
        x_quant = self.scale * (x_int - self.zero_point)

        return x_quant
    
    def _adjust_params_per_axis(self, x_float):
        r = len(x_float.shape)
        new_shape = [1] * self.axis + [-1] + [1] * (r - self.axis -1)
        self._delta = self._delta.reshape(new_shape)
        self._zero_float = self._zero_float.reshape(new_shape)

    def _adjust_params_per_channel(self, x):
        """
        Adjusts the quantization parameter tensors (delta, zero_float)
        to the input tensor shape if they don't match

        Parameters
        ----------
        x: input tensor
        """
        if x.ndim != self.delta.ndim:
            new_shape = [-1] + [1] * (len(x.shape) - 1) #zzk_debug: have problems here. what is the data layout? Should channel be 2?
                                                        # In this code, it seems channel be 3(-1).
            self._delta = self.delta.reshape(new_shape)

            if self._zero_float is not None:
                self._zero_float = self._zero_float.reshape(new_shape)

    def _tensorize_min_max(self, x_min, x_max):
        """
        Converts provided min max range into tensors
        Parameters
        ----------
        x_min: float or Numpy 1D tensor
        x_max: float or Numpy 1D tensor

        Returns
        -------
        x_min: Numpy Tensor 0 or 1-D
        x_max: Numpy Tensor 0 or 1-D
        """
        # Ensure a numpy tensor
        if type(x_min) is not np.ndarray:
            x_min = np.array([x_min])
            x_max = np.array([x_max])
        
        if x_min.ndim == 0:
            x_min = np.array([x_min])
            x_max = np.array([x_max])

        if x_min.ndim > 0 and len(x_min) > 1 and not self.per_channel and self.axis is None:
            raise ValueError(
                'x_min and x_max must be a float or 1-D Tensor'
                ' for per-tensor quantization (per_channel=False)'
            )
        # Ensure we always use zero and avoid division by zero
        x_min = np.minimum(x_min, np.zeros_like(x_min))
        x_max = np.maximum(x_max, np.ones_like(x_max) * self.eps)

        return x_min, x_max

    def set_quant_range(self, x_min, x_max):
        """
        Instantiates the quantization parameters based on the provided
        min and max range

        Parameters
        ----------
        x_min: tensor or float
                Quantization range minimum limit
        x_max: tensor of float
                Quantization range minimum limit
        """
        x_min, x_max = self._tensorize_min_max(x_min, x_max)
        self.float_max = x_max
        self.float_min = x_min
        self._delta = (x_max - x_min) / self.int_max
        self._zero_float = (-x_min / self.delta)
    
    def adjust_per_channel(self, x_float):
        if self.axis is not None:
            self._adjust_params_per_axis(x_float)
        
        if self.per_channel:
            self._adjust_params_per_channel(x_float)


class SymmetricUniformQuantizer(AsymmetricUniformQuantizer):
    """
    Parameters
    ----------
    n_bits: int
        Number of bits for quantization.
    per_channel: bool
        If True: allows for per-channel quantization
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signed = None
    
    @property
    def signed(self):
        assert (self._signed is not None)
        if self._signed is not None:
            return self._signed
        else:
            raise QuantizerNotInitializedError()

    @property
    def symmetric(self):
        return True
    
    @property
    def int_min(self):
        #return -(2.0 ** (self.n_bits - 1)) if self.signed else 0 #zzk_debug: should symmetric quantization use [0,255]?
        return -(2.0 ** (self.n_bits - 1))
    
    @property
    def int_max(self):
        #pos_n_bits = self.n_bits - self.signed
        pos_n_bits = self.n_bits - 1
        return 2.0 ** pos_n_bits - 1
    
    @property
    def zero_point(self):
        return 0.0

    def set_quant_range(self, x_min, x_max):
        x_min, x_max = self._tensorize_min_max(x_min, x_max)
        self.float_max = x_max
        self.float_min = x_min
        self._signed = x_min.min() < 0

        x_absmax = np.maximum(np.abs(x_min), x_max)
        self._delta = x_absmax / self.int_max
    
