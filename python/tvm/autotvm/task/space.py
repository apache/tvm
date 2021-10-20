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
# pylint: disable=too-few-public-methods,invalid-name,unused-argument,arguments-differ
# pylint: disable=consider-using-enumerate,too-many-lines
"""
Template configuration space.

Each template function can be parameterized by a ConfigSpace.
The space is declared when we invoke the template function with ConfigSpace.
During evaluation, we pass in a ConfigEntity, which contains a specific
entity in the space. This entity contains deterministic parameters.
"""
from __future__ import absolute_import as _abs

import itertools
import functools
import math
from collections import namedtuple, OrderedDict
import numpy as np

from tvm.te import schedule, thread_axis
from tvm.tir import expr
from tvm.autotvm.utils import get_const_int

Axis = namedtuple("Axis", ["space", "index"])

try:
    _long = long
except NameError:
    _long = int


class InstantiationError(ValueError):
    """Actively detected error in instantiating a template with a config,
    raised by cfg.raise_error
    e.g. too many unrolling, too many threads in a block
    """


class TransformSpace(object):
    """Base class for transform space
    TransformSpace is the node in the computation graph of axes

    .. note::

        We can regard our schedule code as a transformation graph of axes.
        Starting from raw axes in the definition of te.compute, we can transform these axes
        by some operators. The operator includes 'split', 'reorder' and 'annotate'.
        Each operator has some tunable parameters (e.g. the split factor).
        Then the tuning process is just to find good parameters of these op.

    So all the combinations of the parameters of these op form our search space.

    Naming convention:
    We call the set of all possible values as XXXSpace. (XXX can be Split, Reorder, Config ...)
    We call a specific entity in a space as XXXEntity.
    """

    def __init__(self):
        self.ins = []
        self.num_output = 0
        self.entities = []

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, index):
        """Get an entity of the space by index

        Parameters
        ----------
        index: int

        Returns
        -------
        transform entity
        """
        return self.entities[index]

    @staticmethod
    def get_num_output():
        """get number of output axes after this transform

        Returns
        -------
        n: int
            number of output axes
        """
        return 0


class VirtualAxis(TransformSpace):
    """Axis placeholder in template

    Parameters
    ----------
    var: int or tvm.te.schedule.IterVar
        If is int, return a virtual axis whose length is the provided argument.
        If is IterVar, return a virtual axis whose length is extracted from
        the IterVar's extent domain.

    name: str
    """

    name_ct = 0

    def __init__(self, var, name=None):
        super(VirtualAxis, self).__init__()
        self.num_output = 1

        if name is None:
            name = "axis_%d" % VirtualAxis.name_ct
            VirtualAxis.name_ct += 1

        self.name = name
        if isinstance(var, (int, _long)):
            self.length = var
        elif isinstance(var, schedule.IterVar):
            self.name = var.var.name
            if var.dom is None:
                self.length = -1
            else:
                self.length = get_const_int(var.dom.extent)
        elif isinstance(var, VirtualAxis):
            self.length = var.length
        else:
            raise RuntimeError("Invalid type of axis: " + str(type(var)))

    @staticmethod
    def get_num_output(var, name=None):
        return 1

    def __repr__(self):
        return "vaxis(%s)" % self.name


def get_factors(n):
    """return all factors of an integer

    Parameters
    ----------
    n: int
        integer to factorize

    Returns
    -------
    factors: list
        List of all factors
    """
    step = 2 if n % 2 else 1
    ret = list(
        set(
            functools.reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(math.sqrt(n)) + 1, step) if n % i == 0),
            )
        )
    )
    ret.sort()
    return ret


def get_pow2s(n):
    """return all power-of-two numbers that are less or equal than the integer

    Parameters
    ----------
    n: int
        integer for reference

    Returns
    -------
    factors: list
        List of all power-of-two numbers
    """
    return [2 ** x for x in range(math.floor(math.log2(n)) + 1)]


class SplitSpace(TransformSpace):
    """Split an axis for several times"""

    def __init__(self, axes, policy, **kwargs):
        super(SplitSpace, self).__init__()
        axis = axes[0]

        self.policy = policy
        self.entities = []

        max_factor = kwargs.get("max_factor", 1 << 31)
        fil = kwargs.get("filter", lambda x: True)
        self.product = axis.length
        self.num_output = kwargs.get("num_outputs", 0)
        assert self.num_output > 0

        if policy == "candidate":
            for size in kwargs["candidate"]:
                assert len(size) == self.num_output
                self.entities.append(SplitEntity(size))
        else:
            if policy == "verbose":
                # Include factors and power-of-twos. May generate tails.
                divisibles = get_factors(self.product)
                pow2s = get_pow2s(self.product)
                factors = [x for x in list(set(divisibles) | set(pow2s)) if x <= max_factor]
            elif policy == "factors":
                # Include divisible factors. Guarantee no tails.
                factors = [x for x in get_factors(self.product) if x <= max_factor]
            elif policy == "power2":
                # Include less, equal, and round-up power-of-two numbers. May generate tails.
                factors = [x for x in get_pow2s(self.product) if x <= max_factor]
            else:
                raise RuntimeError("Invalid policy: %s" % policy)

            # Enforce the product of all split factors equals to the axis length
            no_tail = kwargs.get("no_tail", policy == "factors")

            # Generate split entity by enumerating candidate factors.
            self.factors = factors
            self._generate_space(0, [None] * (self.num_output - 1), enforce_no_tail=no_tail)

        self.entities = list(filter(fil, self.entities))

    def _generate_space(self, now, tmp_stack, enforce_no_tail=False):
        """Generate space by DFS"""
        if now == self.num_output - 1:
            prod = functools.reduce(lambda x, y: x * y, tmp_stack)
            if prod > self.product:
                return
            if self.product % prod == 0 or (not enforce_no_tail and prod < self.product):
                self.entities.append(SplitEntity([-1] + tmp_stack[::-1]))
        else:
            for factor in self.factors:
                tmp_stack[now] = factor
                self._generate_space(now + 1, tmp_stack, enforce_no_tail)

    @staticmethod
    def get_num_output(axes, policy, **kwargs):
        return kwargs["num_outputs"]

    def __repr__(self):
        return "Split(policy=%s, product=%d, num_outputs=%d) len=%d" % (
            self.policy,
            self.product,
            self.num_output,
            len(self),
        )


class SplitEntity(object):
    """
    A split operation with detailed parameters
    that can apply to an axis

    Parameters
    ----------
    size: Array of int
        the size of every axis after split.
        e.g. an axis of extent 128, we split it into 3 axes, a possible
        size is [4, 4, 8] (4x4x8 = 128).
    """

    def __init__(self, size):
        self.size = size

    def apply(self, sch, op, axis):
        """Apply split to an axis

        Parameters
        ----------
        sch: tvm.te.schedule.Schedule
            The tvm schedule
        op: tvm.te.Operation
            The stage to be applied
        axis: tvm.te.schedule.IterVar
            axis to split

        Returns
        -------
        axes : list of Axis
            The transformed axes.
        """
        ret = []
        for i in range(1, len(self.size)):
            ax0, ax1 = sch[op].split(axis, int(np.prod(self.size[i:])))
            ret.append(ax0)
            axis = ax1
        return ret + [axis]

    def __repr__(self):
        return str(self.size)


class ReorderSpace(TransformSpace):
    """The parameter space for ordering an array of axes"""

    def __init__(self, axes, policy, **kwargs):
        super(ReorderSpace, self).__init__()
        self.ins = axes
        self.policy = policy
        self.num_output = len(axes)

        if policy == "identity":
            self.entities = [ReorderEntity(range(len(axes)))]
        elif policy == "all":
            self.entities = [ReorderEntity(x) for x in itertools.permutations(range(len(axes)))]
        elif policy == "interval_all":
            begin, end = kwargs["interval"]
            sub_space = list(itertools.permutations(range(begin, end)))
            prefix, suffix = tuple(range(begin)), tuple(range(end, len(axes)))
            self.entities = [ReorderEntity(prefix + x + suffix) for x in sub_space]
        elif policy == "candidate":
            candidate = kwargs["candidate"]
            for can in candidate:
                perm = [axes.index(x) for x in can]
                self.entities.append(ReorderEntity(perm))
        elif policy == "interleave":
            spatial, reduce = kwargs["spatial"], kwargs["reduce"]

            spatial = [[axes.index(x) for x in ch] for ch in spatial]
            reduce = [[axes.index(x) for x in ch] for ch in reduce]

            outer_merged = self._merge_chain([x[:-1] for x in spatial])
            inner_merged = self._merge_chain([x[-1:] for x in spatial] + reduce)

            for o in outer_merged:
                for i in inner_merged:
                    self.entities.append(ReorderEntity(o + i))
        elif policy == "interleave_cuda":
            spatial, reduce = kwargs["spatial"], kwargs["reduce"]

            spatial = [[axes.index(x) for x in ch] for ch in spatial]
            reduce = [[axes.index(x) for x in ch] for ch in reduce]

            outer_merged = self._merge_chain([x[:-1] for x in spatial])
            reduce_merged = self._merge_chain(reduce)
            inner_merged = [x[-1] for x in spatial]

            for o in outer_merged:
                for r in reduce_merged:
                    self.entities.append(ReorderEntity(o + r + inner_merged))
        else:
            raise RuntimeError("Invalid policy: " + policy)

    @staticmethod
    def get_num_output(axes, policy, **kwargs):
        return len(axes)

    def __repr__(self):
        return "Reorder(policy=%s) len=%d" % (self.policy, len(self))

    def _merge_chain(self, chains):
        """generate all combinations of merge some chains"""
        merged = []
        tmp_pt = [0] * len(chains)
        tmp_stack = []

        size = np.sum([len(x) for x in chains])
        self._merge_dfs(chains, size, tmp_pt, tmp_stack, merged)
        return merged

    def _merge_dfs(self, chains, size, tmp_pt, tmp_stack, merged):
        if np.sum(tmp_pt) == size:
            merged.append(list(tmp_stack))
            return

        for i in range(len(chains)):
            # use i == np.argmax(....) here to take spatial order into consideration
            # if we don't want to consider spatial order, we can use tmp_pt[i] == np.max(....)
            if tmp_pt[i] < len(chains[i]) and (
                i == np.argmax([len(chains[x]) - tmp_pt[x] for x in range(len(chains))])
            ):
                tmp_stack.append(chains[i][tmp_pt[i]])
                tmp_pt[i] += 1
                self._merge_dfs(chains, size, tmp_pt, tmp_stack, merged)
                tmp_pt[i] -= 1
                tmp_stack.pop()


class ReorderEntity(object):
    """A reorder operation with detailed parameters that can apply to axes

    Parameters
    ----------
    perm: Array of int
        define the permutation
    """

    def __init__(self, perm):
        self.perm = perm

    def apply(self, sch, op, axes):
        """Apply reorder to an array of axes

        Parameters
        ----------
        sch: tvm.te.schedule.Schedule
            The tvm schedule
        op: tvm.te.Operation
            The stage to be applied
        axis: tvm.te.schedule.IterVar
            axis to split

        Returns
        -------
        axes : list of Axis
            The transformed axes.
        """
        if len(axes) == len(self.perm):
            new_order = [axes[i] for i in self.perm]
        else:
            new_order = [axes[i] for i in self.perm if i < len(axes)]
        sch[op].reorder(*new_order)
        return new_order

    def __repr__(self):
        return str(self.perm)


class AnnotateSpace(TransformSpace):
    """The parameter space for annotating an array of axes"""

    def __init__(self, axes, policy, **kwargs):
        super(AnnotateSpace, self).__init__()

        self.ins = axes
        self.policy = policy
        self.num_output = len(axes)

        if policy == "bind_gpu":
            self.num_axis = len(axes)
            if self.num_axis >= 6:
                self.entities.append(
                    AnnotateEntity(
                        ["fuse"] * (self.num_axis - 6)
                        + [
                            "blockIdx.z",
                            "blockIdx.y",
                            "blockIdx.x",
                            "threadIdx.z",
                            "threadIdx.y",
                            "threadIdx.x",
                        ]
                    )
                )
            elif self.num_axis >= 4:
                self.entities.append(
                    AnnotateEntity(
                        ["fuse"] * (self.num_axis - 4)
                        + ["blockIdx.y", "blockIdx.x", "threadIdx.y", "threadIdx.x"]
                    )
                )
            elif self.num_axis >= 2:
                self.entities.append(
                    AnnotateEntity(["fuse"] * (self.num_axis - 2) + ["blockIdx.x", "threadIdx.x"])
                )
            else:
                raise RuntimeError("Unhandled case in bind_gpu")
        elif policy == "bind_gpu_virtual":
            self.num_axis = len(axes)
            if self.num_axis >= 9:
                self.entities.append(
                    AnnotateEntity(
                        ["fuse"] * (self.num_axis - 9)
                        + [
                            "blockIdx.z",
                            "blockIdx.y",
                            "blockIdx.x",
                            "vthread",
                            "vthread",
                            "vthread",
                            "threadIdx.z",
                            "threadIdx.y",
                            "threadIdx.x",
                        ]
                    )
                )
            elif self.num_axis >= 6:
                self.entities.append(
                    AnnotateEntity(
                        ["fuse"] * (self.num_axis - 6)
                        + [
                            "blockIdx.y",
                            "blockIdx.x",
                            "vthread",
                            "vthread",
                            "threadIdx.y",
                            "threadIdx.x",
                        ]
                    )
                )
            elif self.num_axis >= 3:
                self.entities.append(
                    AnnotateEntity(
                        ["fuse"] * (self.num_axis - 3) + ["blockIdx.x", "vthread", "threadIdx.x"]
                    )
                )
            else:
                raise RuntimeError("Unhandled case in bind_gpu")
        elif policy == "locate_cache":
            self.num_axis = len(axes)
            num_anchor = kwargs["num_anchor"]
            self.anns = list(itertools.combinations(range(self.num_axis), num_anchor))
            self.entities = [AnnotateEntity(x) for x in self.anns]
        else:  # none, vec, unroll, try_vec, try_unroll, try_vec_unroll, ...
            anns = policy.replace("try", "none").split("_")

            for ann in anns:
                if ann not in ["none", "unroll", "vec"]:
                    raise RuntimeError("Invalid policy: " + policy)

            self.num_axis = len(axes)
            self.anns = [anns] * self.num_axis
            self._generate_space(0, [""] * self.num_axis)

    def _generate_space(self, now, tmp_stack):
        """Generate space by DFS"""
        if now == self.num_axis:
            # only vectorize inner most dimension
            vec_ct = tmp_stack.count("vec")
            if vec_ct in (0, 1):
                self.entities.append(AnnotateEntity(list(tmp_stack)))
        else:
            for ann in self.anns[now]:
                tmp_stack[now] = ann
                self._generate_space(now + 1, tmp_stack)

    @staticmethod
    def get_num_output(axes, policy, **kwargs):
        return len(axes)

    def __repr__(self):
        return "Annotate(policy=%s) len=%d" % (self.policy, len(self))


class AnnotateEntity(object):
    """An annotation operation with detailed parameters that can apply to axes

    Parameters
    ----------
    anns: Array of string
        The annotations of axes
    """

    def __init__(self, anns):
        self.anns = anns

    def apply(
        self, sch, op, axes, axis_lens=None, max_unroll=None, vec_size=None, cfg=None, source=None
    ):
        """Apply annotation to an array of axes

        Parameters
        ----------
        sch: tvm.te.schedule.Schedule
            The tvm schedule
        op: tvm.te.Operation
            The stage to be applied
        axes: Array of tvm.te.schedule.IterVar
            axis to split
        axis_lens: Array of int, optional
            the length of axes
        max_unroll: int, optional
            maximum unroll step
        vec_size: Array of int, optional
            valid vector lanes for vectorization
        cfg: ConfigEntity, optional
            cfg for recording error
        source: Array of Array tensor, optional
            source tensor for attaching cache

        Returns
        -------
        axes : list of tvm.te.schedule.IterVar
            The transformed axes
        """
        if source is not None:  # special case : attach cache_read/cache_write
            for src, to in zip(source, self.anns):
                for t in src:
                    sch[t].compute_at(sch[op], axes[to])
        else:  # other cases
            for i, ann in enumerate(self.anns):
                if ann == "none":
                    pass
                elif ann == "unroll":
                    if max_unroll and axis_lens[i] > max_unroll:
                        cfg.raise_error("Too large factor for unrolling")
                    sch[op].unroll(axes[i])
                elif ann == "vec":
                    if vec_size and axis_lens[i] not in vec_size:
                        cfg.raise_error("Wrong size of lanes in vectorization")
                    sch[op].vectorize(axes[i])
                elif ann == "blockIdx.x":
                    sch[op].bind(axes[i], thread_axis("blockIdx.x"))
                elif ann == "blockIdx.y":
                    sch[op].bind(axes[i], thread_axis("blockIdx.y"))
                elif ann == "blockIdx.z":
                    sch[op].bind(axes[i], thread_axis("blockIdx.z"))
                elif ann == "threadIdx.x":
                    sch[op].bind(axes[i], thread_axis("threadIdx.x"))
                elif ann == "threadIdx.y":
                    sch[op].bind(axes[i], thread_axis("threadIdx.y"))
                elif ann == "threadIdx.z":
                    sch[op].bind(axes[i], thread_axis("threadIdx.z"))
                elif ann == "vthread":
                    sch[op].bind(axes[i], thread_axis("vthread"))
                elif ann == "fuse":
                    assert i < len(axes) - 1
                    axes[i + 1] = sch[op].fuse(axes[i], axes[i + 1])
                else:
                    raise RuntimeError("Invalid annotation " + ann)
        return axes

    def __repr__(self):
        return str(self.anns)


class OtherOptionSpace(TransformSpace):
    """The parameter space for general option"""

    def __init__(self, axes, policy, **kwargs):
        super(OtherOptionSpace, self).__init__()

        candidate = kwargs["candidate"]
        self.entities = [OtherOptionEntity(x) for x in candidate]

    @staticmethod
    def get_num_output(axes, policy, **kwargs):
        return 0

    def __repr__(self):
        return "OtherOption(%s) len=%d" % (self.entities, len(self))


class OtherOptionEntity(object):
    """The parameter entity for general option, with a detailed value"""

    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return str(self.val)


class ConfigSpace(object):
    """The configuration space of a schedule. Pass it as config in template to
    collect transformation space and build transform graph of axes
    """

    def __init__(self):
        # private dict to provide sugar
        self.space_map = OrderedDict()  # name -> space
        self._collect = True
        self._length = None
        self._entity_map = OrderedDict()  # name -> entity
        self._constraints = []
        self.errors = []
        self.code_hash = None
        self.flop = 0
        self.cost = None
        self.is_fallback = False

    @staticmethod
    def axis(var):
        """get a virtual axis (axis placeholder)

        Parameters
        ----------
        var: int or tvm.te.schedule.IterVar
            If is int, return an axis whose length is the provided argument.
            If is IterVar, return an axis whose length is extracted from the
            IterVar's extent domain.
        """
        return VirtualAxis(var)

    reduce_axis = axis

    def define_split(self, name, axis, policy="factors", **kwargs):
        """Define a new tunable knob which splits an axis into a list of axes

        Parameters
        ----------
        name: str
            name to index the entity of this space
        axis: tvm.te.schedule.IterVar
            axis to split
        policy: str
            name of policy.
            If is 'factors', the tuner will try all divisible factors.
            If is 'power2', the tuner will try power-of-two factors less or equal to the length.
            If is 'verbose', the tuner will try all candidates in above two policies.
            If is 'candidate', try given candidates.
        **kwargs:
            extra arguments for policy

            ``max_factor``:
                the maximum split factor (`int`).
            ``filter``:
                see examples below for how to use filter (`Callable[[int], bool]`).
            ``num_outputs``:
                the total number of axis after split (`int`).
            ``no_tail``:
                should we only include divisible numbers as split factors (`bool`).
            `candidate``:
                (policy=candidate) manual candidate list (`List`).

        Examples
        --------
        >>> # use custom candidates
        >>> cfg.define_split('tile_x', x, policy='candidate', candidate=[[1, 4, 4], [4, 1, 4]])

        >>> # use a filter that only accepts the split scheme whose inner most tile is less then 4
        >>> cfg.define_split('tile_y', y, policy='factors', filter=lambda x: x.size[-1] <= 4)
        """

        axes = [axis]
        return self._add_new_transform(SplitSpace, name, axes, policy, **kwargs)

    def define_reorder(self, name, axes, policy, **kwargs):
        """Define a new tunable knob which reorders a list of axes

        Parameters
        ----------
        name: str
            name to index the entity of this space
        axes: Array of tvm.te.schedule.IterVar
            axes to reorder
        policy: str
            name of policy
            If is 'identity', do an identity permutation.
            If is 'all', try all permutations.
            If is 'interval_all', try all permutations of an interval of axes.
            If is 'candidate', try listed candidate.
            If is 'interleave', interleave chains of spatial axes and chains of reduction axes.
        kwargs: dict
            extra arguments for policy
        """
        return self._add_new_transform(ReorderSpace, name, axes, policy, **kwargs)

    def define_annotate(self, name, axes, policy, **kwargs):
        """Define a new tunable knob which annotates a list of axes

        Parameters
        ----------
        name: str
            name to index the entity of this space
        axes: Array of tvm.te.schedule.IterVar
            axes to annotate
        policy: str
            name of policy
            If is 'unroll', unroll the axes.
            If is 'try_unroll', try to unroll the axes.
            If is 'try_unroll_vec', try to unroll or vectorize the axes.
            If is 'bind_gpu', bind the first few axes to gpu threads.
            If is 'locate_cache', choose n axes to attach shared/local cache.
        kwargs: dict
            extra arguments for policy
        """
        return self._add_new_transform(AnnotateSpace, name, axes, policy, **kwargs)

    def define_knob(self, name, candidate):
        """Define a tunable knob with a list of candidates

        Parameters
        ----------
        name: str
            name key of that option
        candidate: list
            list of candidates
        """
        return self._add_new_transform(OtherOptionSpace, name, [], None, candidate=candidate)

    def add_flop(self, flop):
        """Add float operation statistics for this tuning task

        Parameters
        ---------
        flop: int or float or IntImm or FloatImm
            number of float operations
        """
        if isinstance(flop, (expr.IntImm, expr.FloatImm)):
            flop = flop.value
        self.flop += float(flop)

    def raise_error(self, msg):
        """register error in config
        Using this to actively detect error when scheduling.
        Otherwise these error will occur during runtime, which
        will cost more time.

        Parameters
        ----------
        msg: str
        """
        self.errors.append(msg)

    def valid(self):
        """Check whether the config meets all the constraints

        .. note::

            This check should be called after instantiation of task,
            because the ConfigEntity/ConfigSpace collects errors during instantiation

        Returns
        -------
        valid: bool
            whether the config meets all the constraints
        """
        return not bool(self.errors)

    def _add_new_transform(self, space_class, name, axes, policy, **kwargs):
        """Add a new transform space in template"""
        # if we do not have tuned info (_collect == True) but defined KNOB value
        # for "default" scheduling before call of _add_new_transform, in this case
        # no need to create new space and override previously pointed KNOB values
        if self._collect and not (self.is_fallback and name in self._entity_map):
            # convert schedule axis to space definition axis
            axes = [x if isinstance(x, (VirtualAxis, Axis)) else self.axis(x) for x in axes]

            # add subspace (knob)
            space = space_class(axes, policy, **kwargs)
            self.space_map[name] = space
            self._entity_map[name] = space[0]
            return [Axis(space, i) for i in range(space.num_output)]
        return [Axis(None, i) for i in range(space_class.get_num_output(axes, policy, **kwargs))]

    def __len__(self):
        if self._length is None:
            self._length = int(np.prod([len(x) for x in self.space_map.values()]))
        return self._length

    def get(self, index):
        """Get a config entity with detailed parameters from this space

        Parameters
        ----------
        index: int
            index in the space
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range: size {}, got index {}".format(len(self), index))
        entities = OrderedDict()
        t = index
        for name, space in self.space_map.items():
            entities[name] = space[t % len(space)]
            t //= len(space)
        ret = ConfigEntity(index, self.code_hash, entities, self._constraints)
        return ret

    def __iter__(self):
        return self._entity_map.__iter__()

    def __getitem__(self, name):
        """get the transform entity(knob) of this entity by name
           do not use this to get a ConfigEntity of this space (should use ConfigSpace.get instead)

        Parameters
        ----------
        name: str
            name of the transform
        """
        return self._entity_map[name]

    def __repr__(self):
        res = "ConfigSpace (len=%d, space_map=\n" % len(self)
        for i, (name, space) in enumerate(self.space_map.items()):
            res += "  %2d %s: %s\n" % (i, name, space)
        return res + ")"


_ann_to_number = {
    "none": 0,
    "vec": 1,
    "unroll": 2,
    "blockIdx.x": 3,
    "blockIdx.y": 4,
    "blockIdx.z": 5,
    "threadIdx.x": 6,
    "threadIdx.y": 7,
    "threadIdx.z": 8,
    "vthread": 9,
    "fuse": 10,
}


class ConfigEntity(ConfigSpace):
    """A configuration with detailed parameters

    Parameters
    ----------
    index: int
        index of this config in space
    code_hash: str
        hash of schedule code
    entity_map: dict
        map name to transform entity
    constraints : list
        List of constraints
    """

    def __init__(self, index, code_hash, entity_map, constraints):
        super(ConfigEntity, self).__init__()
        self.index = index
        self._collect = False
        self._entity_map = entity_map
        self._space_map = None
        self._constraints = constraints
        self.code_hash = code_hash

    def get_flatten_feature(self):
        """flatten entities to a numerical one-dimensional feature vector

        Returns
        -------
        fea: np.array
            one dimensional float32 array
        """
        fea = []
        for _, v in self._entity_map.items():
            if isinstance(v, SplitEntity):
                fea.extend(v.size)
            elif isinstance(v, ReorderEntity):
                # use a naive way: directly copy the permutation
                fea.extend(v.perm)
            elif isinstance(v, AnnotateEntity):
                # one-hot encoding
                for ann in v.anns:
                    tmp = [0] * len(_ann_to_number)
                    tmp[_ann_to_number[ann]] = 1
                    fea.extend(tmp)
            elif isinstance(v, OtherOptionEntity):
                fea.append(v.val)
        return np.array(fea, dtype=np.float32)

    def get_other_option(self):
        """
        Returns
        -------
        other_option: dict
            other tunable parameters (tunable parameters defined by `cfg.define_knob`)
        """
        return {x: x.val for x in self._entity_map.values() if isinstance(x, OtherOptionEntity)}

    def to_json_dict(self):
        """convert to a json serializable dictionary

        Return
        ------
        json_dict: dict
            a json serializable dictionary
        """
        ret = {}
        ret["index"] = int(self.index)
        ret["code_hash"] = self.code_hash
        entity_map = []
        for k, v in self._entity_map.items():
            if isinstance(v, SplitEntity):
                entity_map.append((k, "sp", v.size))
            elif isinstance(v, ReorderEntity):
                entity_map.append((k, "re", v.perm))
            elif isinstance(v, AnnotateEntity):
                entity_map.append((k, "an", v.anns))
            elif isinstance(v, OtherOptionEntity):
                entity_map.append((k, "ot", v.val))
            else:
                raise RuntimeError("Invalid entity instance: " + v)
        ret["entity"] = entity_map
        return ret

    @staticmethod
    def from_json_dict(json_dict):
        """Build a ConfigEntity from json serializable dictionary

        Parameters
        ----------
        json_dict: dict
            Json serializable dictionary. This should be the return value
            of :any:`to_json_dict`.

        Returns
        -------
        config: ConfigEntity
            The corresponding config object

        """
        index = json_dict["index"]
        code_hash = json_dict["code_hash"]
        constraints = []
        entity_map = OrderedDict()

        for item in json_dict["entity"]:
            key, knob_type, knob_args = item
            if knob_type == "sp":
                entity = SplitEntity(knob_args)
            elif knob_type == "re":
                entity = ReorderEntity(knob_args)
            elif knob_type == "an":
                entity = AnnotateEntity(knob_args)
            elif knob_type == "ot":
                entity = OtherOptionEntity(knob_args)
            else:
                raise RuntimeError("Invalid config knob type: " + knob_type)
            entity_map[str(key)] = entity

        return ConfigEntity(index, code_hash, entity_map, constraints)

    def __repr__(self):
        return "%s,%s,%d" % (str(self._entity_map)[12:-1], self.code_hash, self.index)


class FallbackConfigEntity(ConfigSpace):
    """The config entity created to support fallback"""

    def __init__(self):
        super(FallbackConfigEntity, self).__init__()
        self.is_fallback = True

    def fallback_split(self, name, constraints):
        """Fallback a split knob

        Parameters
        ----------
        name: str
            name of the knob
        constraints: List of int
            The maximum tile size for every dimension. Value `-1` means no constraint.

        Examples
        --------
        If you use cfg.define_split('tile_0', 128, num_outputs=3),
        Then cfg.fallback_split('tile_0', [-1, 8, 4]) will give you cfg['tile_0'].size = [4, 8, 4]

        If you use cfg.define_split('tile_0', 49, num_outputs=3),
        Then cfg.fallback_split('tile_0', [-1, 8, 4]) will give you cfg['tile_0'].size = [7, 7, 1]
        """
        space = self.space_map[name]
        assert isinstance(space, SplitSpace)
        assert len(constraints) == space.num_output

        # '-1' means no constraint
        constraints = [x if x != -1 else 1e10 for x in constraints]

        entity = self._entity_map[name]
        now = space.product

        for i in reversed(range(space.num_output)):
            factors = get_factors(now)

            find = len(factors) - 1
            for j, f in enumerate(factors):
                if f > constraints[i]:
                    find = j - 1
                    break

            if find >= 0:
                entity.size[i] = factors[find]
                now //= factors[find]
            else:
                raise RuntimeError("Cannot find feasible fallback split entity for node: " + name)

    def fallback_with_reference_log(self, ref_log):
        """A data driven fallback mechanism.
        We use tuned parameters from TopHub as reference data.
        For an unseen shape, we find the most similar tuned one from TopHub and
        mimic its parameters.
        Note that we are not matching by workload (e.g., input size, kernel size),
        but instead matching by configuration space. The idea is that if two workloads have
        similar configuration space, their optimal configurations are also likely to be similar.

        Parameters
        ----------
        ref_log: List of (autotvm.measure.MeasureInput, autotvm.measure.MeasureResult)
            The reference log
        """
        knob_names = [x for x in self.space_map.keys() if isinstance(self.space_map[x], SplitSpace)]

        # find best match config in reference data by matching tiling factors
        factor_list = []
        for knob_name in knob_names:
            factor_list.append(get_factors(self.space_map[knob_name].product))

        best_match_cfg = None
        best_match_score = 0
        for inp, _ in ref_log:
            match_score = 0
            for i, knob_name in enumerate(knob_names):
                factors = get_factors(int(np.prod(inp.config[knob_name].size)))
                match_score += float(len(set(factor_list[i]).intersection(factors))) / len(
                    factor_list[i]
                )

                if match_score > best_match_score:
                    best_match_score, best_match_cfg = match_score, inp.config

        if best_match_cfg is None:
            return

        # mimic its tiling strategy
        for knob_name in knob_names:
            constraint = list(best_match_cfg[knob_name].size)
            constraint[0] = -1
            self.fallback_split(knob_name, constraint)

        # copy other knobs
        for knob_name in self.space_map.keys():
            if not isinstance(self.space_map[knob_name], SplitSpace):
                self._entity_map[knob_name] = best_match_cfg[knob_name]

    def __setitem__(self, name, entity):
        """set the entity(knob) of by name

        Parameters
        ----------
        name: str
            name of the entity
        entity: SplitEntity, ReorderEntity, AnnotateEntity, OtherOptionEntity
            value of the entity
        """
        self._entity_map[name] = entity

    def __repr__(self):
        return "%s,%s" % (str(self._entity_map)[12:-1], self.code_hash)
