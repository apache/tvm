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

"""
Network implement parameter hiding.
This is like the Pytorch Model, but without the need to declare subnetwork in __init__.
"""
import collections
import numpy as np
from tvm import relay
from tvm.relay import Module
from tvm.relay.prelude import Prelude
from tvm.relay.testing import add_nat_definitions

# pylint: disable=arguments-differ,redefined-builtin,no-else-return,invalid-name
# the orderedset is adopted from ... so I dont think we should change it.
class OrderedSet(collections.MutableSet):
    """
    A set, but keep the elements in the order it was inserted from.
    TODO BEFORE MERGE: INSERT PROPER CITATION
    """
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self):
        key = self.last()
        self.discard(key)
        return key

    def last(self):
        return self.end[1][0]

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

def initialize(param):
    ty = param.type_annotation
    shape = [int(i) for i in ty.shape]
    return np.random.normal(0, 1, shape).astype('float32')

# avoid var duplication
def copy_var(v):
    return relay.Var(v.name_hint, v.type_annotation)

def dedup(l):
    return list(OrderedSet(list(l)))

class Network:
    """
    To define your own network, inherit this class, and define build_impl.
    Call get() to use it when you are done.
    Note that right now this can only emulate inference, as weight are fixed random tensors.
    """
    stack = []
    cnt = 0

    def build_impl(self, **kwargs):
        """
        This is the only function you need to override.
        Inside here, call input()/weight() to generate a input/weight given a relay Var.
        It will return that variable, so you can write `x = self.input(Var("x"))`.
        You can also directly call any instance of Network, including self.
        Additionally, the **kwargs is a dictionary passed in from __init__().
        Use it to define any attribute needed for build_impl().
        return an Expr as the final result of the build_impl().
        """
        raise NotImplementedError

    def __init__(self, *, name="f", **kwargs):
        name = f"{name}_{Network.cnt}"
        Network.cnt += 1
        if len(Network.stack) is not 0:
            mod = Network.stack[-1].mod
            p = Network.stack[-1].p
        else:
            mod = Module()
            p = Prelude(mod)
            add_nat_definitions(p)

        self.mod = mod
        self.p = p
        self.inputs = []
        self.weights = OrderedSet()
        self.sub_network = OrderedSet()
        self.f = relay.GlobalVar(name)
        self.recurse = relay.Var("recurse")
        self.use_recurse = False
        self.ret_type = None
        body = self.build(**kwargs)
        assert isinstance(body, relay.Expr)
        if self.use_recurse:
            inputs = [copy_var(v) for v in self.inputs]
            body = relay.Let(self.recurse,
                             relay.Function(inputs, self.call_from_outside(*inputs)),
                             body)
        self.mod[self.f] = relay.Function(self.inputs + self.all_weights(), body, self.ret_type)

    def build(self, **kwargs):
        Network.stack.append(self)
        try:
            return self.build_impl(**kwargs)
        finally:
            Network.stack.pop()

    def weight(self, w):
        """
        Record the input variable as a weight.
        """
        assert isinstance(w, relay.Var)
        self.weights.add(w)
        return w

    def input(self, i):
        """
        Record the input variable as an input.
        """
        assert isinstance(i, relay.Var)
        self.inputs.append(i)
        return i

    def all_weights(self):
        return dedup(list(self.weights) +
                     [w for n in self.sub_network for w in n.all_weights()])

    def call_from_outside(self, *inputs):
        return self.f(*(list(inputs) + self.all_weights()))

    def __call__(self, *inputs):
        """
        Call the network.

        Parameters
        ----------
          *inputs: List[tvm.relay.Expr]
            The input Expr.

        Returns
        -------
          ret: tvm.relay.Expr
            The result Expr after calling the Network.
        """
        if self in Network.stack:
            self.use_recurse = True
            return self.recurse(*inputs)
        else:
            assert Network.stack
            assert Network.stack[-1].mod == self.mod
            assert Network.stack[-1].p == self.p
            Network.stack[-1].sub_network.add(self)
            return self.call_from_outside(*inputs)

    def interface_type(self):
        """
        Return the type of the expr return from get().
        It will not containts the weights.

        Returns
        -------
          ret: tvm.relay.Type
            The type of get().
        """
        relay.transform.InferType()(self.mod)
        t = self.mod[self.f].checked_type
        return relay.FuncType(t.arg_types[:len(self.inputs)],
                              t.ret_type,
                              t.type_params,
                              t.type_constraints)

    def get(self):
        """
        Return a function with the weights randomly initalized.

        Returns
        -------
          ret: tvm.relay.Function
            A function that wrap the Network.
        """
        weights = []
        for x in self.all_weights():
            ty = x.type_annotation
            assert isinstance(ty, relay.TensorType)
            assert ty.dtype == 'float32'
            shape = [int(i) for i in ty.shape]
            weight = relay.const(np.random.normal(0, 1, shape).astype('float32'))
            weights.append(weight)
        inputs = [copy_var(v) for v in self.inputs]
        return relay.Function(inputs, self.f(*inputs, *weights))
