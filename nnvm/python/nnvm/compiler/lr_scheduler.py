# pylint: disable=too-few-public-methods, no-member
"""API for scheduling learning rate."""
from .. import symbol as sym

class LRScheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    """
    def __init__(self, base_lr=0.01, name='LRScheduler'):
        self.name = name
        self.base_lr = base_lr

    def __call__(self, num_update):
        """Return a new learning rate based on number of updates.

        Parameters
        ----------
        num_update: nnvm Symbol
            the number of updates applied to weight.
        """
        raise NotImplementedError("__call__ method must be overridden.")

class FactorScheduler(LRScheduler):
    """Reduce the learning rate by a factor for every *n* steps.

    It returns a new learning rate by::

        base_lr * pow(factor, num_update/step)

    Parameters
    ----------
    step : int
        Changes the learning rate for every n updates.
    factor : float, optional
        The factor to change the learning rate.
    stop_factor_lr : float, optional
        Stop updating the learning rate if it is less than this value.
    """
    def __init__(self, step, factor=1, stop_factor_lr=1e-8, name='FactorScheduler', **kwargs):
        super(FactorScheduler, self).__init__(name=name, **kwargs)
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr

    def __call__(self, num_update):
        updated_lr = self.base_lr * self.factor ** (num_update / self.step)
        return sym.clip(updated_lr, a_min=self.stop_factor_lr, a_max=self.base_lr)
