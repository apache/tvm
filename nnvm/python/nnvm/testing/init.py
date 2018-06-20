"""Initializer of parameters."""
import numpy as np

class Initializer(object):
    """The base class of an initializer."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self, desc, arr):
        """Initialize an array

        Parameters
        ----------
        desc : str
            Initialization pattern descriptor.

        arr : NDArray
            The array to be initialized.
        """
        if desc.endswith('weight'):
            self._init_weight(desc, arr)
        elif desc.endswith('bias'):
            self._init_bias(desc, arr)
        elif desc.endswith('gamma'):
            self._init_gamma(desc, arr)
        elif desc.endswith('beta'):
            self._init_beta(desc, arr)
        elif desc.endswith('mean'):
            self._init_mean(desc, arr)
        elif desc.endswith('var'):
            self._init_var(desc, arr)
        else:
            self._init_default(desc, arr)

    def _init_bias(self, _, arr):
        arr[:] = 0.0

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_mean(self, _, arr):
        arr[:] = 0.0

    def _init_var(self, _, arr):
        arr[:] = 1.0

    def _init_weight(self, name, arr):
        """Abstract method to Initialize weight."""
        raise NotImplementedError("Must override it")

    def _init_default(self, name, _):
        raise ValueError(
            'Unknown initialization pattern for %s. ' \
            'Default initialization is now limited to '\
            '"weight", "bias", "gamma" (1.0), and "beta" (0.0).' \
            'Please use mx.sym.Variable(init=mx.init.*) to set initialization pattern' % name)


class Xavier(Initializer):
    """ "Xavier" initialization for weights

    Parameters
    ----------
    rnd_type: str, optional
        Random generator type, can be ``'gaussian'`` or ``'uniform'``.

    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    magnitude: float, optional
        Scale of random number.
    """
    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super(Xavier, self).__init__(rnd_type=rnd_type,
                                     factor_type=factor_type,
                                     magnitude=magnitude)
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def _init_weight(self, name, arr):
        shape = arr.shape
        hw_scale = 1.
        if len(shape) < 2:
            raise ValueError('Xavier initializer cannot be applied to vector {0}. It requires at'
                             ' least 2D.'.format(name))
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        factor = 1.
        if self.factor_type == "avg":
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == "in":
            factor = fan_in
        elif self.factor_type == "out":
            factor = fan_out
        else:
            raise ValueError("Incorrect factor type")
        # Hack for mobilenet, because there is less connectivity
        if "depthwise" in name:
            factor = 3 * 3
        scale = np.sqrt(self.magnitude / factor)
        if self.rnd_type == "uniform":
            arr[:] = np.random.uniform(-scale, scale, size=arr.shape)
        else:
            raise ValueError("Unknown random type")
