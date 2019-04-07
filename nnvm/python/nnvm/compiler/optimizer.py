# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, no-member, too-few-public-methods, too-many-arguments, too-many-locals, protected-access
"""Optimizer API"""
from . import graph_util
from .. import symbol as sym

class Optimizer(object):
    """Base class inherited by all optimizers.

    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate.

    lr_scheduler : LRScheduler, optional
        The learning rate scheduler.

    rescale_grad : float, optional
        Multiply the gradient with `rescale_grad` before updating. Often
        choose to be ``1.0/batch_size``.

    clip_gradient : float, optional
        Clip the gradient by projecting onto the box ``[-clip_gradient, clip_gradient]``.

    wd : float, optional
        The weight decay (or L2 regularization) coefficient. Modifies objective
        by adding a penalty for having large weights.

    name : string, optional
        The name of optimizer.
    """
    def __init__(self, learning_rate=0.01, lr_scheduler=None,
                 rescale_grad=1, clip_gradient=None, wd=0, name="Optimizer"):
        self.name = name
        self.lr = learning_rate
        self.lr_scheduler = lr_scheduler
        self.rescale_grad = rescale_grad
        self.clip_gradient = clip_gradient
        self.wd = wd
        init_update_t = sym.Variable(name+'_t', init=sym.zeros(shape=(1,), dtype="int32"))
        self.update_t = sym._assign(init_update_t, init_update_t + 1)

    def minimize(self, obj, var=None):
        """Minimize given obj symbol respect to var. If var is not set, all input
        variables of obj will be used.

        Parameters
        ----------
        obj : nnvm Symbol or list of nnvm Symbols
            Symbols to be minimized.
        var : nnvm Symbol or list of nnvm Symbols, optional
            Symbols the gradient respect to.

        Returns
        -------
        group_sym : nnvm Symbol
            Group symbol represents update symbols.
        """
        raise NotImplementedError()

    def _get_lr(self):
        """Gets the learning rate with learning rate scheduler.

        Returns
        -------
        lr : float
            Learning rate.
        """
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.update_t)
        else:
            lr = self.lr
        return lr


class SGD(Optimizer):
    """The SGD optimizer
    """
    def __init__(self, name='SGD', **kwargs):
        super(SGD, self).__init__(name=name, **kwargs)

    def minimize(self, obj, var=None):
        variables = var or obj.list_input_variables()
        if not isinstance(variables, list):
            variables = [variables]
        grads = graph_util.gradients(obj, variables)
        updates = []
        lr_t = self._get_lr()
        for v, g in zip(variables, grads):
            g = self.rescale_grad * g
            if self.clip_gradient is not None:
                g = sym.clip(g, a_min=-1 * self.clip_gradient, a_max=self.clip_gradient)
            updates.append(sym._assign(v, v - lr_t * (g + self.wd * v)))
        return sym.Group(updates)


class Adam(Optimizer):
    """The Adam optimizer.

    This class implements the optimizer described in *Adam: A Method for
    Stochastic Optimization*, available at http://arxiv.org/abs/1412.6980.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, name='Adam', **kwargs):
        super(Adam, self).__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []

    def minimize(self, obj, var=None):
        variables = var or obj.list_input_variables()
        if not isinstance(variables, list):
            variables = [variables]
        grads = graph_util.gradients(obj, variables)
        updates = []
        for i, v in enumerate(variables):
            self.m.append(sym.Variable(self.name + '_m' + str(i), init=sym.zeros_like(v)))
            self.v.append(sym.Variable(self.name + '_v' + str(i), init=sym.zeros_like(v)))
        rate = sym.sqrt(1 - self.beta2 ** self.update_t) / (1 -  self.beta1 ** self.update_t)
        lr_t = self._get_lr() * rate
        for variable, g, m, v in zip(variables, grads, self.m, self.v):
            g = self.rescale_grad * g
            if self.clip_gradient is not None:
                g = sym.clip(g, a_min=-1 * self.clip_gradient, a_max=self.clip_gradient)
            update_m = sym._assign(m, self.beta1 * m + (1 - self.beta1) * g)
            update_v = sym._assign(v, self.beta2 * v + (1 - self.beta2) * g * g)
            update_var = sym._assign(variable, variable - lr_t * (update_m / (sym.sqrt(update_v) \
                         + self.epsilon) + self.wd * variable))
            updates.append(update_var)
        return sym.Group(updates)
