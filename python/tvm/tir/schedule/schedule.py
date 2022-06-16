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
"""The TensorIR schedule class"""
from typing import Callable, Dict, List, Optional, Tuple, Union

from tvm._ffi import register_object as _register_object
from tvm.error import TVMError, register_error
from tvm.ir import IRModule, PrimExpr
from tvm.runtime import Object, String
from tvm.tir import Block, Buffer, FloatImm, For, IntImm, PrimFunc

from ..function import IndexMap
from . import _ffi_api
from ._type_checker import type_checked
from .state import ScheduleState, StmtSRef, _parse_debug_mask, _parse_mod
from .trace import Trace


@register_error
class ScheduleError(TVMError):
    """Error that happens during TensorIR scheduling."""


@_register_object("tir.LoopRV")
class LoopRV(Object):
    """A random variable that refers to a loop"""

    def __init__(self) -> None:
        """Construct a new LoopRV."""
        self.__init_handle_by_constructor__(
            _ffi_api.LoopRV  # type: ignore # pylint: disable=no-member
        )


@_register_object("tir.BlockRV")
class BlockRV(Object):
    """A random variable that refers to a block"""

    def __init__(self) -> None:
        """Construct a new BlockRV."""
        self.__init_handle_by_constructor__(
            _ffi_api.BlockRV  # type: ignore # pylint: disable=no-member
        )


# It is a workaround for mypy: https://github.com/python/mypy/issues/7866#issuecomment-549454370
# This feature is not supported until python 3.10:
# https://docs.python.org/3.10/whatsnew/3.10.html#pep-613-typealias
ExprRV = Union[PrimExpr]  # A random variable that evaluates to an integer

RAND_VAR_TYPE = Union[ExprRV, BlockRV, LoopRV]  # pylint: disable=invalid-name

# Update to `Literal["detail", "fast", "none"]` once upgraded to python3.8
_ERROR_RENDER_LEVEL: Dict[str, int] = {
    "detail": 0,
    "fast": 1,
    "none": 2,
}


def _parse_error_render_level(error_render_level: str) -> int:
    if error_render_level not in _ERROR_RENDER_LEVEL:
        raise ValueError(
            'error_render_level can be "detail", "fast", or "none", but got: '
            + f"{error_render_level}"
        )
    return _ERROR_RENDER_LEVEL.get(error_render_level)


def _parse_seed(seed: Optional[int]) -> int:
    if seed is None:
        return -1
    if not isinstance(seed, int):
        raise TypeError(f"Expected `seed` to be int or None, but gets: {seed}")
    if seed < 1 or seed > 2147483647:
        raise ValueError(f"seed must be in the range [1, 2147483647], but gets: {seed}")
    return seed


@_register_object("tir.Schedule")
class Schedule(Object):
    """The user-facing schedule class

    A schedule is a set of transformations that change the order of computation but
    preserve the semantics of computation. Some example of schedules:
    1) Split a loop into two;
    2) Reorder two loops;
    3) Inline the computation of a specific buffer into its consumer

    The schedule class stores auxiliary information to schedule correctly and efficiently.

    Link to tutorial: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
    """

    @type_checked
    def __init__(
        self,
        mod: Union[PrimFunc, IRModule],
        *,
        seed: Optional[int] = None,
        debug_mask: Union[str, int] = "none",
        error_render_level: str = "detail",
    ) -> None:
        """Construct a TensorIR schedule class from an IRModule

        Parameters
        ----------
        mod : Union[PrimFunc, IRModule]
            The IRModule or PrimFunc to be scheduled
        seed: Optional[int]
            The seed value for schedule's random state
            Note that None and -1 means use device random, otherwise only integer between 1 and
            2147483647 is allowed.
        debug_mask : Union[str, int]
            Do extra correctness checking after the class creation and each time
            after calling the Replace method.
            Possible choices of `debug_mask`:
            1) "all" - Turn on all the checks
            2) "none" - Turn off all the checks
            3) An integer - Turn on checks according to the bitmasks provided in ScheduleDebugMask
        error_render_level : str = "detail"
            The level of error rendering. Choices: "detail", "fast", "none".
            - "detail": Render a detailed error message, with the TIR and error locations printed
            - "fast: Show a simple error message without rendering or string manipulation
            - "none": Do not show any error message.

        Note
        ----
        The checks performed includes:
        1) VerifySRefTree
        2) VerifyCachedFlags
        """
        # call the constructor
        self.__init_handle_by_constructor__(
            _ffi_api.TracedSchedule,  # type: ignore # pylint: disable=no-member
            _parse_mod(mod),
            _parse_seed(seed),
            _parse_debug_mask(debug_mask),
            _parse_error_render_level(error_render_level),
        )

    @staticmethod
    def _create_non_traced(
        mod: Union[PrimFunc, IRModule],
        *,
        seed: Optional[int] = None,
        debug_mask: Union[str, int] = "none",
        error_render_level: str = "detail",
    ) -> "Schedule":
        """Construct a non-traced TensorIR schedule class from an IRModule."""
        return _ffi_api.ConcreteSchedule(  # type: ignore # pylint: disable=no-member
            _parse_mod(mod),
            _parse_seed(seed),
            _parse_debug_mask(debug_mask),
            _parse_error_render_level(error_render_level),
        )

    ########## Utilities ##########

    @property
    def mod(self) -> IRModule:
        """Returns the AST of the module being scheduled"""
        return _ffi_api.ScheduleGetMod(self)  # type: ignore # pylint: disable=no-member

    @property
    def state(self) -> ScheduleState:
        """Returns the ScheduleState in the current schedule class"""
        return _ffi_api.ScheduleGetState(self)  # type: ignore # pylint: disable=no-member

    @property
    def trace(self) -> Optional[Trace]:
        """Returns the internally maintained trace of scheduling program execution"""
        return _ffi_api.ScheduleGetTrace(self)  # type: ignore # pylint: disable=no-member

    def copy(self) -> "Schedule":
        """Returns a copy of the schedule, including both the state and the symbol table,
        * guaranteeing that
        * 1) SRef tree is completely reconstructed;
        * 2) The IRModule being scheduled is untouched;
        * 3) All the random variables are valid in the copy, pointing to the corresponding sref
        * reconstructed

        Returns
        -------
        copy : Schedule
            A new copy of the schedule
        """
        return _ffi_api.ScheduleCopy(self)  # type: ignore # pylint: disable=no-member

    @type_checked
    def seed(self, seed: int) -> None:
        """Seed the randomness

        Parameters
        ----------
        seed : int
            The new random seed, -1 if use device random, otherwise non-negative
        """
        return _ffi_api.ScheduleSeed(self, seed)  # type: ignore # pylint: disable=no-member

    def fork_seed(self) -> int:
        """Returns a forked random state as seed for new schedules

        Returns
        -------
        seed : int
            The forked random state, not the same as the current random state
        """
        return _ffi_api.ScheduleForkSeed(self)  # type: ignore # pylint: disable=no-member

    @type_checked
    def show(self, rand_var: RAND_VAR_TYPE) -> str:
        """Returns a string representation of the value that the random variable evaluates to

        Parameters
        ----------
        rand_var : Union[ExprRV, BlockRV, LoopRV]
            The random variable to be evaluated

        Returns
        -------
        str_repr : str
            The string representation
        """
        return str(self.get(rand_var))

    ########## Lookup ##########

    @type_checked
    def get(
        self,
        rand_var_or_sref: Union[RAND_VAR_TYPE, StmtSRef],
    ) -> Optional[Union[int, Block, For]]:
        """Returns:
        - the corresponding Block that a BlockRV evaluates to;
        - the corresponding For that a LoopRV evaluates to;
        - the corresponding integer that a ExprRV evaluates to;
        - the corresponding Block that a block sref points to;
        - the corresponding For that a loop sref points to;

        Parameters
        ----------
        rand_var_or_sref : Union[ExprRV, BlockRV, LoopRV, StmtSRef]
            The random variable / sref to be evaluated

        Returns
        -------
        result : Optional[Union[int, Block, For]]
            The corresponding result
        """
        if isinstance(rand_var_or_sref, StmtSRef):
            return rand_var_or_sref.stmt
        result = _ffi_api.ScheduleGet(self, rand_var_or_sref)  # type: ignore # pylint: disable=no-member
        if isinstance(result, IntImm):
            result = result.value
        return result

    @type_checked
    def get_sref(self, rand_var_or_stmt: Union[BlockRV, LoopRV, Block, For]) -> Optional[StmtSRef]:
        """Returns the corresponding sref to the given
        1) LoopRV
        2) BlockRV
        3) Block
        4) For

        Parameters
        ----------
        rand_var_or_stmt : Union[BlockRV, LoopRV, Block, For]
            The random variable / sref to be evaluated

        Returns
        -------
        result : Optional[StmtSRef]
            The corresponding result
        """
        return _ffi_api.ScheduleGetSRef(  # type: ignore # pylint: disable=no-member
            self, rand_var_or_stmt
        )

    @type_checked
    def remove_rv(self, rand_var: RAND_VAR_TYPE) -> None:
        """Remove a random variable from the symbol table

        Parameters
        ----------
        rand_var : Union[BlockRV, LoopRV, ExprRV]
            The random variable to be removed
        """
        return _ffi_api.ScheduleRemoveRV(self, rand_var)  # type: ignore # pylint: disable=no-member

    ########## Schedule: Sampling ##########

    @type_checked
    def sample_categorical(
        self,
        candidates: List[int],
        probs: List[float],
        decision: Optional[int] = None,
    ) -> ExprRV:
        """Sample an integer given the probability distribution

        Parameters
        ----------
        candidates : List[int]
            The candidates to be sampled from
        probs : List[float]
            The probability of each candidate
        decision : Optional[int]
            The sampling decision, if any

        Returns
        -------
        result : ExprRV
            The random variable sampled from candidates
        """
        return _ffi_api.ScheduleSampleCategorical(  # type: ignore # pylint: disable=no-member
            self,
            candidates,
            probs,
            decision,
        )

    @type_checked
    def sample_perfect_tile(
        self,
        loop: LoopRV,
        n: int,
        max_innermost_factor: int = 16,
        decision: Optional[List[int]] = None,
    ) -> List[ExprRV]:
        """Sample the factors to perfect tile a specific loop

        Parameters
        ----------
        loop : LoopRV
            The loop to be tiled
        n : int
            The number of tiles to be sampled
        max_innermost_factor : int
            The maximum tile size allowed to be sampled in the innermost loop
        decision: Optional[List[int]]
            The sampling decision, if any

        Returns
        -------
        result : List[ExprRV]
            A list of length `n`, the random perfect tile sizes sampled
        """
        return list(
            _ffi_api.ScheduleSamplePerfectTile(  # type: ignore  # pylint: disable=no-member
                self,
                loop,
                n,
                max_innermost_factor,
                decision,
            )
        )

    @type_checked
    def sample_compute_location(
        self,
        block: Union[BlockRV, str],
        decision: Optional[int] = None,
    ) -> LoopRV:
        """Sample a compute-at location of the given block

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block whose compute-at location is to be sampled
        decision : Optional[int]
            The sampling decision

        Returns
        -------
        result : LoopRV
            The sampled loop where the input block is to be computed at
        """
        block = self._normalize_block_arg(block)

        return _ffi_api.ScheduleSampleComputeLocation(  # type: ignore  # pylint: disable=no-member
            self,
            block,
            decision,
        )

    ########## Schedule: Get blocks & loops ##########
    @type_checked
    def get_block(
        self,
        name: str,
        func_name: str = "main",
    ) -> BlockRV:
        """Retrieve a block in a specific function with its name

        Parameters
        ----------
        name : str
            The name of the block
        func_name : str = "main"
            The name of the function

        Returns
        -------
        block : BlockRV
            The block retrieved
            IndexError is raised if 0 or multiple blocks exist with the specific name.
        """
        return _ffi_api.ScheduleGetBlock(  # type: ignore # pylint: disable=no-member
            self,
            name,
            func_name,
        )

    @type_checked
    def get_loops(self, block: Union[BlockRV, str]) -> List[LoopRV]:
        """Get the parent loops of the block in its scope, from outer to inner

        Parameters
        ----------
        block : Union[BlockRV, str]
            The query block

        Returns
        -------
        loops : List[LoopRV]
            A list of loops above the given block in its scope, from outer to inner
        """
        block = self._normalize_block_arg(block)
        return list(_ffi_api.ScheduleGetLoops(self, block))  # type: ignore # pylint: disable=no-member

    @type_checked
    def get_child_blocks(self, block_or_loop: Union[BlockRV, LoopRV]) -> List[BlockRV]:
        """Get the leaf blocks of a specific block/loop

        Parameters
        ----------
        block_or_loop : Union[BlockRV, LoopRV]
            The query block/loop

        Returns
        -------
        blocks : List[LoopRV]
            A list of leaf blocks inside a specific block/loop
        """
        return list(_ffi_api.ScheduleGetChildBlocks(self, block_or_loop))  # type: ignore # pylint: disable=no-member

    @type_checked
    def get_producers(self, block: Union[BlockRV, str]) -> List[BlockRV]:
        """Get the producers of a specific block

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block in the query

        Returns
        -------
        producers : List[BlockRV]
            A list of producers of the given block
        """
        block = self._normalize_block_arg(block)
        return list(_ffi_api.ScheduleGetProducers(self, block))  # type: ignore # pylint: disable=no-member

    @type_checked
    def get_consumers(self, block: Union[BlockRV, str]) -> List[BlockRV]:
        """Get the consumers of a specific block

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block in the query

        Returns
        -------
        consumers : List[BlockRV]
            A list of consumers of the given block
        """
        block = self._normalize_block_arg(block)
        return list(_ffi_api.ScheduleGetConsumers(self, block))  # type: ignore # pylint: disable=no-member

    ########## Schedule: Transform loops ##########
    @type_checked
    def fuse(
        self,
        *loops: List[LoopRV],
        preserve_unit_iters: bool = True,
    ) -> LoopRV:
        """Fuse a list of consecutive loops into one. It requires:
        1) The loops can't have annotations or thread bindings.
        2) The (i+1)-th loop must be the only child of the i-th loop.
        3) All loops must start with 0.
        4) The domain of a loop to be fused cannot depend on another loop to be fused.

        Parameters
        ----------
        *loops : List[LoopRV]
            The loops to be fused

        Returns
        -------
        fused_loop : LoopRV
            The new loop after fusion

        Examples
        --------

        Before applying fuse, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_fuse(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do fuse:

        .. code-block:: python

            sch = tir.Schedule(before_fuse)
            i, j = sch.get_loops(sch.get_block("B"))
            sch.fuse(i, j)
            print(sch.mod["main"].script())

        After applying fuse, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_fuse(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                # the 2 loops are fused into 1
                for i_j_fused in T.serial(0, 16384):
                    with T.block("B"):
                        vi = T.axis.S(128, T.floordiv(i_j_fused, 128))
                        vj = T.axis.S(128, T.floormod(i_j_fused, 128))
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        return _ffi_api.ScheduleFuse(self, loops, preserve_unit_iters)  # type: ignore # pylint: disable=no-member

    @type_checked
    def split(
        self,
        loop: LoopRV,
        factors: List[Union[int, ExprRV, None]],
        preserve_unit_iters: bool = True,
    ) -> List[LoopRV]:
        """Split a loop into a list of consecutive loops. It requires:
        1) The loop can't have annotation or thread binding.
        2) The loop must start with 0.
        Predicates may be added to ensure the total loop numbers keeps unchanged.
        In `factors`, at most one of the factors can be None,
        which will be automatically inferred.

        Parameters
        ----------
        loop : LoopRV
            The loop to be split

        factors: List[Union[int, ExprRV, None]]
            The splitting factors
            Potential inputs are:
            - None
            - ExprRV
            - Positive constant integers

        preserve_unit_iters : bool
            Whether or not to preserve unit iterators in block bindings

        Returns
        -------
        split_loops : List[LoopRV]
            The new loops after split

        Examples
        --------

        Before split, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_split(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do split:

        .. code-block:: python

            sch = tir.Schedule(before_split)
            i, j = sch.get_loops(sch.get_block("B"))
            sch.split(i, factors=[2, 64])
            print(sch.mod["main"].script())

        After applying split, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_split(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                # the original loop is split into 2 loops
                for i0, i1, j in T.grid(2, 64, 128):
                    with T.block("B"):
                        vi = T.axis.S(128, i0 * 64 + i1)
                        vj = T.axis.S(128, j)
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        # it will be checked later in C++ implementation
        # that there is at most one None in `factors`
        return list(
            _ffi_api.ScheduleSplit(  # type: ignore # pylint: disable=no-member
                self,
                loop,
                factors,
                preserve_unit_iters,
            )
        )

    @type_checked
    def reorder(self, *ordered_loops: List[LoopRV]) -> None:
        """
        Reorder a list of loops. It doesn't require the loops to be consecutive.
        It requires:
        1) The loops are in the same chain. That means: the loops can be ordered to [l_1, l_2, ... ,
        l_n] where l_i is an ancestor of l_{i+1} and there are only single-branch loops between
        l_1 and l_n (which also indicates they are under the same scope).
        2) After reordering, the domain of an outer loop cannot depend on any of the inner loops.
        3) For every block under the loop nests, its block binding must be affine, and the block
        variables must be either data parallel or reduction.
        4) No duplicated loops are allowed in the arguments.

        Parameters
        ----------
        *ordered_loops : List[LoopRV]
            The loops in the new order

        Examples
        --------

        Before reorder, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_reorder(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do reorder:

        .. code-block:: python

            sch = tir.Schedule(before_reorder)
            i, j = sch.get_loops(sch.get_block("B"))
            sch.reorder(j, i)
            print(sch.mod["main"].script())

        After applying reorder, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_reorder(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                # Here j and i are reordered
                for j, i in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleReorder(self, ordered_loops)  # type: ignore # pylint: disable=no-member

    @type_checked
    def add_unit_loop(self, block_or_loop: Union[LoopRV, BlockRV]) -> LoopRV:
        """Create a new unit loop on top of the specific block or loop.

        Parameters
        ----------
        block_or_loop : Union[LoopRV, BlockRV]
            The block above which the new loop is created

        Returns
        -------
        new_loop : LoopRV
            The new unit loop

        Examples
        --------

        Before add_unit_loop, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_add_unit_loop(
                A: T.Buffer[(), "int32"],
                B: T.Buffer[(), "int32"],
                C: T.Buffer[(), "int32"],
            ) -> None:
                with T.block("C"):
                    vi = T.axis.spatial(1, 0)
                    C[()] = A[()] + B[()]

        Create the schedule and do add-unit-loop:

        .. code-block:: python

            sch = tir.Schedule(before_add_unit_loop)
            sch.add_unit_loop(sch.get_block("C"))
            print(sch.mod["main"].script())

        After applying add-unit-loop, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_add_unit_loop(
                A: T.Buffer[(), "int32"],
                B: T.Buffer[(), "int32"],
                C: T.Buffer[(), "int32"],
            ) -> None:
                for u in T.serial(1):
                    with T.block("C"):
                        vi = T.axis.spatial(1, 0)
                        C[()] = A[()] + B[()]
        """
        return _ffi_api.ScheduleAddUnitLoop(self, block_or_loop)  # type: ignore # pylint: disable=no-member

    ########## Schedule: Manipulate ForKind ##########

    @type_checked
    def parallel(self, loop: LoopRV) -> None:
        """Parallelize the input loop. It requires:
        1) The scope block that the loop is in should have stage-pipeline property
        2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
        bindings
        3) For each block under the loop, the loop can only be contained in data-parallel block
        iters' bindings

        Parameters
        ----------
        loop : LoopRV
            The loop to be parallelized

        Examples
        --------

        Before parallel, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_parallel(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do parallel:

        .. code-block:: python

            sch = tir.Schedule(before_parallel)
            i, j = sch.get_loops(sch.get_block("B"))
            sch.parallel(i)

        After applying parallel, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_parallel(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i in T.parallel(0, 128):
                    for j in T.serial(0, 128):
                        with T.block("B"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleParallel(self, loop)  # type: ignore # pylint: disable=no-member

    @type_checked
    def vectorize(self, loop: LoopRV) -> None:
        """Vectorize the input loop. It requires:
        1) The scope block that the loop is in should have stage-pipeline property
        2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
        bindings
        3) For each block under the loop, the loop can only be contained in data-parallel block
        iters' bindings

        Parameters
        ----------
        loop : LoopRV
            The loop to be vectorized

        Examples
        --------

        Before vectorize, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_vectorize(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do vectorize:

        .. code-block:: python

            sch = tir.Schedule(before_vectorize)
            i, j = sch.get_loops(sch.get_block("B"))
            sch.vectorize(j)

        After applying vectorize, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_vectorize(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i in T.serial(0, 128):
                    for j in T.vectorized(0, 128):
                        with T.block("B"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleVectorize(self, loop)  # type: ignore # pylint: disable=no-member

    @type_checked
    def bind(self, loop: LoopRV, thread_axis: str) -> None:
        """Bind the input loop to the given thread axis. It requires:
        1) The scope block that the loop is in should have stage-pipeline property
        2) All the blocks under the loop are complete blocks or reduction blocks, and have affine
        bindings
        3) For each block under the loop, if the thread axis starts with "threadIdx`, the loop can
        only be contained in data-parallel block iter and reduction block iters' bindings. Otherwise
        the loop can only be contained in data-parallel block iters' bindings

        Parameters
        ----------
        loop : LoopRV
            The loop to be bound to the thread axis
        thread_axis : str
            The thread axis to be bound to the loop. Possible candidates:
            - blockIdx.x/y/z
            - threadIdx.x/y/z
            - vthread.x/y/z
            - vthread (It is a legacy behavior that will be deprecated. Please use `vthread.x/y/z`
            instead.)

        Examples
        --------

        Before bind, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_bind(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do bind:

        .. code-block:: python

            sch = tir.Schedule(before_bind)
            i, j = sch.get_loops(sch.get_block("B"))
            sch.bind(i, "blockIdx.x")
            sch.bind(j, "threadIdx.x")

        After applying bind, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_bind(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i in T.thread_binding(0, 128, thread = "blockIdx.x"):
                    for j in T.thread_binding(0, 128, thread = "threadIdx.x"):
                        with T.block("B"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleBind(self, loop, thread_axis)  # type: ignore # pylint: disable=no-member

    @type_checked
    def unroll(self, loop: LoopRV) -> None:
        """Unroll the input loop. It requires nothing

        Parameters
        ----------
        loop : LoopRV
            The loop to be unrolled

        Examples
        --------

        Before unroll, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_unroll(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do unroll:

        .. code-block:: python

            sch = tir.Schedule(before_unroll)
            i, j = sch.get_loops(sch.get_block("B"))
            sch.unroll(i)

        After applying unroll, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_unroll(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i in T.unroll(0, 128):
                    for j in T.serial(0, 128):
                        with T.block("B"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleUnroll(self, loop)  # type: ignore # pylint: disable=no-member

    ########## Schedule: Insert cache stages ##########

    @type_checked
    def cache_read(
        self, block: Union[BlockRV, str], read_buffer_index: int, storage_scope: str
    ) -> BlockRV:
        """Create a block that reads a buffer region into a read cache. It requires:

        1) There is at most one block who write the buffer in the scope.

        2) The scope block have stage-pipeline property.

        Parameters
        ----------
        block : Union[BlockRV, str]
            The consumer block of the target buffer.

        read_buffer_index: int
            The index of the buffer in block's read region.

        storage_scope: str
            The target storage scope.

        Returns
        -------
        cached_block : BlockRV
            The block of the cache stage

        Examples
        --------
        Before cache_read, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_cache_read(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and cache_read:

        .. code-block:: python

            sch = tir.Schedule(before_cache_read)
            block_b = sch.get_block("B")
            sch.cache_read(block_b, 0, "local")
            print(sch.mod["main"].script())

        After applying cache_read, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_cache_read(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                A_local = T.alloc_buffer((128, 128), scope="local")
                for i, j in T.grid(128, 128):
                    with T.block("A_local"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        A_local[vi, vj] = A[vi, vj]
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A_local[vi, vj] * 2.0

        """
        block = self._normalize_block_arg(block)
        return _ffi_api.ScheduleCacheRead(  # type: ignore # pylint: disable=no-member
            self, block, read_buffer_index, storage_scope
        )

    @type_checked
    def cache_write(
        self, block: Union[BlockRV, str], write_buffer_index: int, storage_scope: str
    ) -> BlockRV:
        """Create a block that reads a buffer region into a write cache. It requires:

        1) There is only one block who write the buffer in the scope.

        2) The scope block have stage-pipeline property.

        Parameters
        ----------
        block : Union[BlockRV, str]
            The producer block of the target buffer.

        write_buffer_index: int
            The index of the buffer in block's write region.

        storage_scope: str
            The target storage scope.


        Returns
        -------
        cached_block : BlockRV
            The block of the cache stage

        Examples
        --------
        Before cache_write, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_cache_write(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and cache_write:

        .. code-block:: python

            sch = tir.Schedule(before_cache_write)
            block_b = sch.get_block("B")
            sch.cache_write(block_b, 0, "local")
            print(sch.mod["main"].script())

        After applying cache_write, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_cache_write(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                B_local = T.alloc_buffer((128, 128), scope="local")
                for i, j in T.grid(128, 128):
                    with T.block("A_local"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_local[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = B_local[vi, vj]

        """
        block = self._normalize_block_arg(block)
        return _ffi_api.ScheduleCacheWrite(  # type: ignore # pylint: disable=no-member
            self, block, write_buffer_index, storage_scope
        )

    @type_checked
    def reindex(
        self,
        block: Union[BlockRV, str],
        buffer: Union[Tuple[str, int], str, Buffer],
    ) -> BlockRV:
        """Create a block that read/write a buffer region into a read/write cache with reindexing.
        The layout of the cache will be the same as by the iterators of the block that reads/writes
        the buffer. It requires:
        1) There is only one block who reads/writes the target buffer
        2) There is only one buffer load/store of this buffer in the block

        Parameters
        ----------
        block : Union[BlockRV, str]

            The block that accesses the target buffer.  If a string,
            this must uniquely identify a block.

        buffer: Union[Tuple[str,int], Buffer, str]

            The buffer to be transformed, or a specification of how to
            identify the buffer to be transformed.

            If `buffer` if a tuple of ``(str,int)``, the first item
            should be either "read" or "write", and the second item is
            an index into the block's read or write regions.

            If `buffer` is a string, it is the name of the buffer,
            which must exist within the reads/writes of the block.  In
            addition, the reads/writes of the block may not contain
            more than one buffer with this name.

            If `buffer` is a Buffer object, it must exist within the
            reads/writes of the block.

        Returns
        -------
        reindex_block : BlockRV
            The block of the reindex stage

        Examples
        --------

        Before transform_layout, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_reindex(
                A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"]
            ) -> None:
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vj, vi] * 2.0

        Create the schedule and do transform_layout:

        .. code-block:: python

            sch = tir.Schedule(before_reindex)
            block = sch.get_block("B")
            sch.reindex(block, ("read", 0))

        After applying reindex, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_reindex(
                A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"]
            ) -> None:
                A_reindex = T.alloc_buffer((128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with T.block("A_reindex"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        A_reindex[vi, vj] = A[vj, vi]
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A_reindex[vi, vj] * 2.0

        """
        block = self._normalize_block_arg(block)
        buffer_index_type, buffer_index, _ = self._normalize_buffer_arg(block, buffer)
        assert buffer_index_type in ["read", "write"], "Invalid buffer_index_type"
        buffer_index_type_enum = 0 if buffer_index_type == "read" else 1
        return _ffi_api.ScheduleReIndex(  # type: ignore # pylint: disable=no-member
            self, block, buffer_index, buffer_index_type_enum
        )

    ########## Schedule: Compute location ##########

    @type_checked
    def compute_at(
        self,
        block: Union[BlockRV, str],
        loop: LoopRV,
        preserve_unit_loops: bool = False,
    ) -> None:
        """Compute-At. Move a producer block under the specific loop, and regenerate the
        loops induced by the block so that the buffer region produced by the producer block could
        cover those regions consumed by its consumer blocks under the given loop. It requires:

        1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`

        2) The scope block has stage-pipeline property

        3) The subtree of the scope block, where the given block is in, satisfies the compact
        dataflow condition. i.e. all the blocks in the scope block's subtree must be either
        complete block or reduction block

        4) The block is not an output block with regard to the scope block, i.e. the buffers written
        by the block are allocated under the scope block

        5) All the consumers of the block are under the given loop

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block to be moved

        loop: LoopRV
            The loop where the block to be moved under

        preserve_unit_loops: bool
            Whether to keep the trivial loops whose extents are 1

        Examples
        --------

        Before compute-at, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_compute_at(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128), "float32")
                B = T.alloc_buffer((128, 128), "float32")
                C = T.match_buffer(c, (128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do compute-at:

        .. code-block:: python

            sch = tir.Schedule(before_compute_at)
            block = sch.get_block("B")
            loop, _ = sch.get_loops(sch.get_block("C"))
            sch.compute_at(block, loop, preserve_unit_loops=False)
            print(sch.mod["main"].script())

        After applying compute-at, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_compute_at(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128), "float32")
                B = T.alloc_buffer((128, 128), "float32")
                C = T.match_buffer(c, (128, 128), "float32")
                for i in T.serial(0, 128):
                    for j in T.serial(0, 128):
                        with T.block("B"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0
                    for j in T.serial(0, 128):
                        with T.block("C"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            C[vi, vj] = B[vi, vj] + 1.0

        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleComputeAt(  # type: ignore # pylint: disable=no-member
            self,
            block,
            loop,
            preserve_unit_loops,
        )

    @type_checked
    def reverse_compute_at(
        self,
        block: Union[BlockRV, str],
        loop: LoopRV,
        preserve_unit_loops: bool = False,
    ) -> None:
        """Reverse-Compute-At. Move a consumer block under the specific loop, and regenerate the
        loops induced by the block so that the buffer region consumed by the consumer block could
        cover those regions produced by its producer blocks under the given loop. It requires:

        1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`

        2) The scope block has stage-pipeline property

        3) The subtree of the scope block, where the given block is in, satisfies the compact
        dataflow condition. i.e. all the blocks in the scope block's subtree must be either
        complete block or reduction block

        4) All the producers of the block are under the given loop

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block to be moved

        loop: LoopRV
            The loop where the block to be moved under

        preserve_unit_loops: bool
            Whether to keep the trivial loops whose extents are 1

        Examples
        --------

        Before reverse-compute-at, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_reverse_compute_at(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128), "float32")
                B = T.alloc_buffer((128, 128), "float32")
                C = T.match_buffer(c, (128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do reverse-compute-at:

        .. code-block:: python

            sch = tir.Schedule(before_reverse_compute_at)
            block = sch.get_block("C")
            loop, _ = sch.get_loops(sch.get_block("B"))
            sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
            print(sch.mod["main"].script())

        After applying reverse-compute-at, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_reverse_compute_at(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128), "float32")
                B = T.alloc_buffer((128, 128), "float32")
                C = T.match_buffer(c, (128, 128), "float32")
                for i in T.serial(0, 128):
                    for j in T.serial(0, 128):
                        with T.block("B"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0
                    for j in T.serial(0, 128):
                        with T.block("C"):
                            vi, vj = T.axis.remap("SS", [i, j])
                            C[vi, vj] = B[vi, vj] + 1.0

        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleReverseComputeAt(  # type: ignore # pylint: disable=no-member
            self,
            block,
            loop,
            preserve_unit_loops,
        )

    @type_checked
    def compute_inline(self, block: Union[BlockRV, str]) -> None:
        """Inline a block into its consumer(s). It requires:

        1) The block is a complete non-root block, which only produces one buffer

        2) The block must not be the only leaf in the scope.

        3) The body of the block must be a BufferStore statement in
           the form of, ``A[i, j, k, ...] = ...`` where the indices of
           the LHS are all distinct atomic variables, and no variables
           other than those indexing variables are allowed in the
           statement.

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block to be inlined to its consumer(s)

        Examples
        --------

        Before compute-inline, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_inline(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.alloc_buffer((128, 128))
                C = T.match_buffer(c, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do compute-inline:

        .. code-block:: python

            sch = tir.Schedule(before_inline)
            sch.compute_inline(sch.get_block("B"))
            print(sch.mod["main"].script())

        After applying compute-inline, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_inline(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                C = T.match_buffer(c, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = A[vi, vj] * 2.0 + 1.0

        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleComputeInline(self, block)  # type: ignore # pylint: disable=no-member

    @type_checked
    def reverse_compute_inline(self, block: Union[BlockRV, str]) -> None:
        """Inline a block into its only producer. It requires:

        1) The block is a complete non-root block, which only produces and consumes one buffer

        2) The block must not be the only leaf in the scope.

        3) The only producer of the block is a read-after-write producer and a
           complete non-root block

        4) The body of the block must be a BufferStore statement in the form of,
           ``B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)`` where the
           indices of each `BufferLoad` on the RHS are all distinct atomic
           variables, and no variables other than those indexing variables are
           allowed in the statement.

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block to be inlined to its producer

        Examples
        --------

        Before reverse-compute-inline, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_inline(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.alloc_buffer((128, 128))
                C = T.match_buffer(c, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do reverse-compute-inline:

        .. code-block:: python

            sch = tir.Schedule(before_inline)
            sch.reverse_compute_inline(sch.get_block("C"))
            print(sch.mod["main"].script())

        After applying reverse-compute-inline, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_inline(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                C = T.match_buffer(c, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = A[vi, vj] * 2.0 + 1.0

        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleReverseComputeInline(self, block)  # type: ignore # pylint: disable=no-member

    ########## Schedule: Reduction ##########

    @type_checked
    def decompose_reduction(self, block: Union[BlockRV, str], loop: LoopRV) -> BlockRV:
        """Decompose a reduction block into two separate blocks.

        a) The init block, which is translated from the init statement of the reduction block;

        b) The update block, which is the original block without init statement.

        The init block is inserted right before the given loop.

        The schedule primitive requires:

        1) The input block is a reduction block.

        2) The input loop is the ancestor of the block.

        3) The input loop is not lower than all the loops related to reduce block var.

        Parameters
        ----------
        block : Union[BlockRV, str]
            The reduction block to be decomposed
        loop : LoopRV
            The loop above which the init block is inserted before.

        Returns
        -------
        init_block : BlockRV
            The init block

        Examples
        --------
        Before decompose-reduction, in TensorIR, the IR is:

        .. code-block:: python

            @tvm.script.tir
            def before_decompose(a: ty.handle, c: ty.handle) -> None:
                A = tir.match_buffer(a, [128, 128])
                B = tir.match_buffer(b, [128, 128])
                C = tir.match_buffer(c, [128, 128])
                for i, j, k in tir.grid(128, 128, 128):
                    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
                        with tir.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        Create the schedule and do decompose-reduction with specified loop:

        .. code-block:: python

            sch = tir.Schedule(before_decompose)
            C = sch.get_block("C")
            i, j, k = sch.get_loops(C)
            sch.decompose_reduction(C, i)
            print(tvm.script.asscript(sch.mod["main"]))

        After applying decompose-reduction, the IR becomes:

        .. code-block:: python

            @tvm.script.tir
            def after_decompose(a: ty.handle, c: ty.handle) -> None:
                A = tir.match_buffer(a, [128, 128])
                B = tir.match_buffer(b, [128, 128])
                C = tir.match_buffer(c, [128, 128])
                for i in tir.serial(128):
                    for j in tir.serial(128):
                        with tir.block([128, 128]) as [vi, vj]:
                            C[vi, vj] = 0.0
                for i, j, k in tir.grid(128, 128, 128):
                    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        """
        block = self._normalize_block_arg(block)
        return _ffi_api.ScheduleDecomposeReduction(self, block, loop)  # type: ignore # pylint: disable=no-member

    @type_checked
    def rfactor(self, loop: LoopRV, factor_axis: int) -> LoopRV:
        """Factorize an associative reduction block by the specified loop.

        An associative reduction cannot be parallelized directly,
        because it leads to potential race condition during accumulation.
        Alternatively, the reduction could be factorized on a loop with the following steps:
        - Step 1: evenly slice the reduction into `n` separate chunks, where `n` is the loop extent
        - Step 2: compute the chunks separately and write the result into `n` intermediate buffers;
        - Step 3: accumulate the `n` separate buffer into the result buffer.
        Note that the Step 2 above introduces opportunities for parallelization.

        RFactor is a schedule primitive that implements the transformation described above:
        Given a block that writes to buffer `B`, it factorizes a loop of extent `n`.

        For example, the pseudocode below accumulates `B[i] = sum(A[i, : , : ])`:

        .. code-block:: python

            for i in range(128):                    # loop i is a data parallel loop
                for j in range(128):                # loop j is a reduction loop
                    for k in range(128):            # loop k is a reduction loop
                        B[i] = B[i] + A[i, j, k]

        Suppose RFactor is applied on the innermost loop `k` and `factor_axis = 1`.
        RFactor then creates an intermediate buffer and two blocks.

        1. The intermediate buffer, or "rf-buffer" is a buffer of rank `ndim(B) + 1` and
        size `size(B) * n`, whose shape expands from `shape(B)` by adding an axis of `n`
        at the position specified by `factor_axis`. For example,

            * shape(B) = [1, 2, 3], factor_axis = 0  => shape(B_rf) = [n, 1, 2, 3]
            * shape(B) = [1, 2, 3], factor_axis = 1  => shape(B_rf) = [1, n, 2, 3]
            * shape(B) = [1, 2, 3], factor_axis = 2  => shape(B_rf) = [1, 2, n, 3]
            * shape(B) = [1, 2, 3], factor_axis = 3  => shape(B_rf) = [1, 2, 3, n]

        2. The rfactor block, or "rf-block", is a block that writes to the `rf-buffer` without
        accumulating over the loop `k`, i.e. the loop `k` is converted from a reduction loop
        to a data parallel loop. In our example, the rf-block is:

        .. code-block:: python

            B_rf = np.zeros((128, 128))     # the rf-buffer
            for k in range(128):            # loop k is converted to a data parallel loop
                for i in range(128):        # loop i is a data parallel loop (unchanged)
                    for j in range(128):    # loop j is a reduction loop (unchanged)
                        B_rf[i, k] = B_rf[i, k] + A[i, j, k]


        3. The write-back block, or `wb-block`, is a block that accumulates the rf-buffer into
        the result buffer. All the reduction loops are removed except the loop `k` for accumulation.
        In our example, the wb-block is:

        .. code-block:: python

            for i in range(128):            # loop i is a data parallel loop (unchanged)
                                            # loop j is removed because it is a reduction loop
                for k in range(128):        # loop k is a reduction loop (unchanged)
                    B[i] = B[i] + B_rf[i, k]


        Parameters
        ----------
        loop : LoopRV
            The loop outside block for which we want to do rfactor
        factor_axis : int
            The position where the new dimension is placed in the new introduced rfactor buffer

        Returns
        -------
        rf_block : BlockRV
            The block which computes partial results over each slices (i.e., the first block
            as described in the above illustration)

        Examples
        --------

        Before rfactor, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_rfactor(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128, 128))
                B = T.match_buffer(b, (128,))
                for ii, i, j in T.grid(128, 128, 128):
                with T.block("B"):
                    vii, vi, vj = T.axis.remap("SRR", [ii, i, j])
                    with T.init():
                        B[vii] = 0.0
                    B[vii] = B[vii] + A[vii, vi, vj]

        Create the schedule and do rfactor:

        .. code-block:: python

            sch = tir.Schedule(before_rfactor)
            _, _, k = sch.get_loops(sch.get_block("B"))
            sch.rfactor(k, 0)
            print(sch.mod["main"].script())

        After applying rfactor, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_rfactor(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, [128, 128, 128])
                B = T.match_buffer(b, [128])
                B_rf = T.alloc_buffer([128, 128])
                for i2, ii, i in T.grid(128, 128, 128):
                    with T.block("B_rf"):
                        vi2, vii, vi = T.axis.remap("SSR", [i2, ii, i])
                        with T.init():
                            B_rf[vi2, vii] = 0.0
                        B_rf[vi2, vii] = (B_rf[vi2, vii] + A[vii, vi, vi2])
                for ii, i2 in T.grid(128, 128):
                    with T.block("B"):
                        vii, vi2 = T.axis.remap("SR", [ii, i2])
                        with T.init():
                            B[vii] = 0.0
                        B[vii] = B[vii] + B_rf[vi2, vii]


        Note
        ----

        Rfactor requires:
        1) `loop` has only one child block, and it is a reduction block;
        2) `loop` is a reduction loop, i.e. the loop variable is bound to only reduction variables
        in the block binding;
        3) `loop` is not parallelized, vectorized, unrolled or bound to any thread axis;
        4) The block scope that `loop` is in is a staged-pipeline;
        5) The outermost loop outside the reduction block should has the reduction block as its
        first child block;
        6) The outermost reduction loop should have only one child block;
        7) An unary extent loop that is not bound to any reduction or data parallel variables in
        the block binding should not appear under some reduction loop;
        8) The reduction block should write to only one buffer, and its init and body are both
        simple `BufferStore`s, and the pattern is registered as an associative reducer.
        The pre-defined patterns include: plus, multiplication, min and max;
        9) Each of the loops on top of the block cannot be bound to a data parallel and a
        reduction block binding at the same time;
        10) `factor_axis` should be in range `[-ndim(B) - 1, ndim(B)]`,
        where `B` is the buffer that the reduction block writes to.
        Negative indexing is normalized according to numpy convention.
        """
        return _ffi_api.ScheduleRFactor(self, loop, factor_axis)  # type: ignore # pylint: disable=no-member

    ######## Schedule: Block annotation ########

    @type_checked
    def storage_align(  # pylint: disable=too-many-arguments
        self,
        block: Union[BlockRV, str],
        buffer_index: int,
        axis: int,
        factor: int,
        offset: int,
    ) -> None:
        """Set alignment requirement for specific dimension such that
        stride[axis] == k * factor + offset for some k. This is useful to set memory layout for more
        friendly memory access pattern. For example, we can set alignment to be factor=2, offset=1
        to avoid bank conflict for thread access on higher dimension in GPU shared memory.

        Parameters
        ----------
        block : Union[BlockRV, str]
            The producer block of the buffer.
        buffer_index : int
            The index of the buffer in block's write region.
        axis : int
            The dimension to be specified for alignment.
        factor : int
            The factor multiple of alignment.
        offset : int
            The required offset factor.

        Examples
        --------

        Before storage_align, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_storage_align(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.alloc_buffer((128, 128))
                C = T.match_buffer(c, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do storage_align:

        .. code-block:: python

            sch = tir.Schedule(before_storage_align)
            sch.storage_align(sch.get_block("B"), buffer_index=0, axis=0, factor=128, offset=1)
            print(sch.mod["main"].script())

        After applying storage_align, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_storage_align(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.alloc_buffer((128, 128))
                C = T.match_buffer(c, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        T.block_attr({"buffer_dim_align": [[[0, 128, 1]]]})
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        After lowering passes, buffer B will have strides as [129, 1].

        Note
        ----
        Storage_align requires the buffer to be an intermediate buffer defined via `alloc_buffer`.
        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleStorageAlign(  # type: ignore # pylint: disable=no-member
            self, block, buffer_index, axis, factor, offset
        )

    @type_checked
    def set_scope(self, block: Union[BlockRV, str], buffer_index: int, storage_scope: str) -> None:
        """Set the storage scope of a buffer, where the buffer is
        specified by the a block and a write-index

        Parameters
        ----------
        block : Union[BlockRV, str]
            The producer block of the buffer
        buffer_index : int
            The index of the buffer in block's write region
        storage_scope : str
            The storage scope to be set

        Examples
        --------

        Before set_scope, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_set_scope(
                A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]
            ) -> None:
                B = T.alloc_buffer((128, 128), dtype="float32")

                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do set_scope:

        .. code-block:: python

            sch = tir.Schedule(before_set_scope)
            sch.set_scope(sch.get_block("B"), buffer_index=0, storage_scope="shared")
            print(sch.mod["main"].script())

        After applying set_scope, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_set_scope(
                A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]
            ) -> None:
                B_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")

                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B_shared[vi, vj] = A[vi, vj] * T.float32(2)
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B_shared[vi, vj] + T.float32(1)

        Note
        ----
        Set_scope requires the buffer to be an intermediate buffer defined via `alloc_buffer`.
        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleSetScope(  # type: ignore # pylint: disable=no-member
            self, block, buffer_index, storage_scope
        )

    ########## Schedule: Blockize & Tensorize ##########

    @type_checked
    def blockize(self, loop: LoopRV) -> BlockRV:
        """Convert the subtree rooted at a specific loop into a block.

        Parameters
        ----------
        loop : LoopRV
            The root of the subtree.

        Returns
        -------
        result : BlockRV
            The new block.

        Examples
        --------

        Before blockize, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_blockize(
                A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"]
            ) -> None:
                for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
                    with T.block("B"):
                        vi = T.axis.spatial(128, i_0 * 16 + i_1)
                        vj = T.axis.spatial(128, j_0 * 16 + j_1)
                        T.reads(A[vi, vj])
                        T.writes(B[vi, vj])
                        B[vi, vj] = A[vi, vj] * T.float32(2)

        Create the schedule and do set_scope:

        .. code-block:: python

            sch = tir.Schedule(before_blockize)
            B = sch.get_block("B")
            _, _, i1, _ = sch.get_loops(B)
            sch.blockize(i1)
            print(sch.mod["main"].script())

        After applying blockize, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_blockize(
                A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"]
            )-> None:
                for i_0, j_0 in T.grid(8, 8):
                    with T.block("B_o"):
                        vio, vjo = T.axis.remap("SS", [i_0, j_0])
                        T.reads(A[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
                        T.writes(B[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
                        for i_1, j_1 in T.grid(16, 16):
                            with T.block("B"):
                                vi, vj = T.axis.remap("SS", [i_1, j_1])
                                T.reads(A[vio * 16 + vi, vjo * 16 + vj])
                                T.writes(B[vio * 16 + vi, vjo * 16 + vj])
                                B[vio * 16 + vi, vjo * 16 + vj] = A[vio * 16 + vi, vjo * 16 + vj] \
                                                                  * T.float32(2)

        Note
        ----
        blockize requires there is exactly one block under the given loop and the bindings of the
        block are divisible by the subspace represented by the loops starting at the given loop.
        """

        return _ffi_api.ScheduleBlockize(self, loop)  # type: ignore # pylint: disable=no-member

    @type_checked
    def tensorize(self, block_or_loop: Union[BlockRV, LoopRV], tensor_intrin: str) -> None:
        """Tensorize the computation enclosed by loop with the tensor intrinsic.

        Parameters
        ----------
        block_or_loop : Union[BlockRV, LoopRV]
            The loop to be tensorized.
        tensor_intrin : str
            The tensor intrin or the name of the tensor intrin.

        Examples
        --------

        Before tensorize, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_tensorize(
                A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"],
            ) -> None:
                # body
                # with T.block("root")
                for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(8, 8, 8, 16, 16, 16):
                    with T.block("update"):
                        vi = T.axis.spatial(128, i_0 * 16 + i_1)
                        vj = T.axis.spatial(128, j_0 * 16 + j_1)
                        vk = T.axis.reduce(128, k_0 * 16 + k_1)
                        T.reads(C[vi, vj], A[vi, vk], B[vj, vk])
                        T.writes(C[vi, vj])
                        with T.init():
                            C[vi, vj] = T.float32(0)
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        Declare and register the tensor intrinsic:

        .. code-block:: python

            @T.prim_func
            def mma_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
                B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
                C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

                with T.block("root"):
                    T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
                    T.writes(C[0 : 16, 0 : 16])
                    for i, j, k in T.grid(16, 16, 16):
                        with T.block("update"):
                            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


            @T.prim_func
            def mma_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (16, 16), align=128, offset_factor=1)
                B = T.match_buffer(b, (16, 16), align=128, offset_factor=1)
                C = T.match_buffer(c, (16, 16), align=128, offset_factor=1)

                with T.block("root"):
                    T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
                    T.writes(C[0 : 16, 0 : 16])
                    T.evaluate(
                        T.tvm_mma_sync(
                            C.data,
                            C.elem_offset // 256,
                            A.data,
                            A.elem_offset // 256,
                            B.data,
                            B.elem_offset // 256,
                            C.data,
                            C.elem_offset // 256,
                            dtype="handle",
                        )
                    )

            tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)

        Create the schedule and do tensorize:

        .. code-block:: python

            sch = tir.Schedule(before_tensorize)
            update = sch.get_block("update")
            _, _, _, i1, _, _ = sch.get_loops(update)
            sch.tensorize(i1, "test_mma_intrin")
            print(sch.mod["main"].script())

        After applying tensorize, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_tensorize(
                A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"],
            ) -> None:
                # body
                # with T.block("root")
                for i_0, j_0, k_0 in T.grid(8, 8, 8):
                    with T.block("update_o"):
                        vio, vjo, vko = T.axis.remap("SSR", [i_0, j_0, k_0])
                        T.reads(
                            C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                            A[vio * 16 : vio * 16 + 16, vko * 16 : vko * 16 + 16],
                            B[vjo * 16 : vjo * 16 + 16, vko * 16 : vko * 16 + 16],
                        )
                        T.writes(C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
                        A_1 = T.match_buffer(
                            A[vio * 16 : vio * 16 + 16, vko * 16 : vko * 16 + 16],
                            [16, 16],
                            dtype="float32",
                            offset_factor=1,
                        )
                        B_1 = T.match_buffer(
                            B[vjo * 16 : vjo * 16 + 16, vko * 16 : vko * 16 + 16],
                            [16, 16],
                            dtype="float32",
                            offset_factor=1,
                        )
                        C_1 = T.match_buffer(
                            C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                            [16, 16],
                            dtype="float32",
                            offset_factor=1,
                        )
                        with T.init():
                            for i_1, j_1 in T.grid(16, 16):
                                with T.block("update_init"):
                                    vi_init, vj_init = T.axis.remap("SS", [i_1, j_1])
                                    T.reads()
                                    T.writes(C[vio * 16 + vi_init, vjo * 16 + vj_init])
                                    C[vio * 16 + vi_init, vjo * 16 + vj_init] = T.float32(0)
                        T.evaluate(
                            T.tvm_mma_sync(
                                C_1.data,
                                C_1.elem_offset // 256,
                                A_1.data,
                                A_1.elem_offset // 256,
                                B_1.data,
                                B_1.elem_offset // 256,
                                C_1.data,
                                C_1.elem_offset // 256,
                                dtype="handle",
                            )
                        )
        """
        _ffi_api.ScheduleTensorize(  # type: ignore # pylint: disable=no-member
            self, block_or_loop, tensor_intrin
        )

    ########## Schedule: Annotation ##########

    @type_checked
    def annotate(
        self,
        block_or_loop: Union[BlockRV, LoopRV],
        ann_key: str,
        ann_val: Union[str, int, float, ExprRV, List[Union[str, int, float, ExprRV]]],
    ) -> None:
        """Annotate a block/loop with a key value pair

        Parameters
        ----------
        block_or_loop: Union[BlockRV, LoopRV]
            The block/loop to be annotated
        ann_key : str
            The annotation key
        ann_val : Union[str, int, float, ExprRV, List[Union[str, int, float, ExprRV]]]
            The annotation value

        Examples
        --------

        Before annotate, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_annotate(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do annotate:

        .. code-block:: python

            sch = tir.Schedule(before_annotate)
            sch.annotate(sch.get_block("B"), "ann_key", "ann_value")
            print(sch.mod["main"].script())

        After applying annotate, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_annotate(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.block_attr({"ann_key", "ann_value"})
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        if isinstance(ann_val, str):
            ann_val = String(ann_val)
        elif isinstance(ann_val, int):
            ann_val = IntImm("int32", ann_val)
        elif isinstance(ann_val, float):
            ann_val = FloatImm("float32", ann_val)
        _ffi_api.ScheduleAnnotate(  # type: ignore # pylint: disable=no-member
            self, block_or_loop, ann_key, ann_val
        )

    @type_checked
    def unannotate(self, block_or_loop: Union[BlockRV, LoopRV], ann_key: str) -> None:
        """Unannotate a block/loop's annotation with key ann_key

        Parameters
        ----------
        block_or_loop: Union[BlockRV, LoopRV]
            The block/loop to be unannotated
        ann_key : str
            The annotation key

        Examples
        --------

        Before unannotate, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_unannotate(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.block_attr({"ann_key", "ann_value"})
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do annotate:

        .. code-block:: python

            sch = tir.Schedule(before_unannotate)
            sch.unannotate(sch.get_block("B"), "ann_key")
            print(sch.mod["main"].script())

        After applying unannotate, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_unannotate(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (128, 128))
                B = T.match_buffer(b, (128, 128))
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleUnannotate(  # type: ignore # pylint: disable=no-member
            self, block_or_loop, ann_key
        )

    ########## Schedule: Layout transformation ##########

    def _normalize_block_arg(self, block: Union[BlockRV, str]) -> BlockRV:
        if isinstance(block, str):
            return self.get_block(block)

        return block

    def _normalize_buffer_arg(
        self, block: BlockRV, buffer: Union[Tuple[str, int], str, Buffer]
    ) -> Tuple[str, int, Buffer]:

        block_name = self.get(block).name_hint

        def iter_buffers():
            block_obj = self.get(block)
            for i, read in enumerate(block_obj.reads):
                yield "read", i, read.buffer
            for i, write in enumerate(block_obj.writes):
                yield "write", i, write.buffer

        if isinstance(buffer, str):
            possible_buffers = {}
            # String lookup requires ensuring that the name is unique
            for buffer_index, buffer_index_type, buf in iter_buffers():
                if buf.name == buffer:
                    possible_buffers[buf] = (buffer_index_type, buffer_index)

            assert possible_buffers, f"Could not find buffer '{buffer}' in block '{block_name}'"
            assert (
                len(possible_buffers) == 1
            ), f"Multiple buffers named '{buffer}' in block '{block_name}'"
            buffer_obj, (buffer_index, buffer_index_type) = next(iter(possible_buffers.items()))

        elif isinstance(buffer, Buffer):
            # Buffer lookup has unique id, can break out early
            found = False
            for buffer_index, buffer_index_type, buffer_obj in iter_buffers():
                if buffer_obj.same_as(buffer):
                    found = True
                    break

            assert found, "Could not find buffer '{buffer.name}' in block '{block_name}'"

        elif isinstance(buffer, tuple):
            buffer_index_type, buffer_index = buffer
            assert buffer_index_type in ["read", "write",], (
                f"Invalid buffer_index_type.  "
                f"Expected 'read' or 'write', "
                f"but received {buffer_index_type}"
            )
            buffer_list = (
                self.get(block).reads if buffer_index_type == "read" else self.get(block).writes
            )
            assert 0 <= buffer_index < len(buffer_list), (
                f"Invalid buffer_index {buffer_index}.  "
                f"Block {block_name} has only "
                f"{len(buffer_list)} {buffer_index_type} buffers."
            )
            buffer_obj = buffer_list[buffer_index].buffer

        else:
            raise TypeError(f"Invalid type for argument 'buffer': {type(buffer)}")

        return (buffer_index_type, buffer_index, buffer_obj)

    @type_checked
    def transform_layout(
        self,
        block: Union[BlockRV, str],
        buffer: Union[Tuple[str, int], str, Buffer],
        index_map: Union[IndexMap, Callable],
    ) -> None:
        """Apply a transformation represented by IndexMap to buffer

        Parameters
        ----------
        block : Union[BlockRV, str]

            The block that accesses the target buffer.  If a string,
            this must uniquely identify a block.

        buffer: Union[Tuple[str,int], Buffer, str]

            The buffer to be transformed, or a specification of how to
            identify the buffer to be transformed.

            If `buffer` if a tuple of ``(str,int)``, the first item
            should be either "read" or "write", and the second item is
            an index into the block's read or write regions.

            If `buffer` is a string, it is the name of the buffer,
            which must exist within the reads/writes of the block.  In
            addition, the reads/writes of the block may not contain
            more than one buffer with this name.

            If `buffer` is a Buffer object, it must exist within the
            reads/writes of the block.

        index_map : Union[IndexMap, Callable]

            The transformation to apply.

            If `index_map` is a callable, and the returned list
            contains IndexMap.AXIS_SEPARATOR, the SetAxisSeparators
            primitive will be called in addition to the
            TransformLayout primitive.

        Examples
        --------
        Before transform_layout, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_transform_layout(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128), "float32")
                B = T.alloc_buffer((128, 128), "float32")
                C = T.match_buffer(c, (128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do transform_layout:

        .. code-block:: python

            sch = tir.Schedule(before_storage_align)
            sch.transform_layout(sch.get_block("B"), buffer=("write",0),
                                 index_map=lambda m, n: (m // 16, n // 16, m % 16, n % 16))
            print(sch.mod["main"].script())

        After applying transform_layout, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def two_elementwise_transformed_intermediate_buffer(a: T.handle, c: T.handle) -> None:
                A = T.match_buffer(a, (128, 128), "float32")
                B = T.alloc_buffer((8, 8, 16, 16), "float32")
                C = T.match_buffer(c, (128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0

        """
        block = self._normalize_block_arg(block)
        buffer_index_type, buffer_index, buffer_obj = self._normalize_buffer_arg(block, buffer)

        ndim = len(buffer_obj.shape)
        if callable(index_map):
            index_map, axis_separators = IndexMap.from_func_with_separators(index_map, ndim=ndim)
        else:
            axis_separators = []

        buffer_index_type_enum = 0 if buffer_index_type == "read" else 1
        _ffi_api.ScheduleTransformLayout(  # type: ignore # pylint: disable=no-member
            self, block, buffer_index, buffer_index_type_enum, index_map
        )
        if axis_separators:
            _ffi_api.ScheduleSetAxisSeparator(  # type: ignore # pylint: disable=no-member
                self, block, buffer_index, buffer_index_type_enum, axis_separators
            )

    @type_checked
    def transform_block_layout(
        self,
        block: Union[BlockRV, str],
        index_map: Union[IndexMap, Callable],
    ) -> None:
        """Apply a transformation represented by IndexMap to block

        Parameters
        ----------
        block : Union[BlockRV, str]
            The block to be transformed

        index_map : Union[IndexMap, Callable]
            The transformation to apply.

        Examples
        --------

        Before transform_block_layout, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_transform_block_layout(
                A: T.Buffer[(16, 16), "float32"],
                B: T.Buffer[(16, 16), "float32"]
            ) -> None:
                for i, j in T.grid(16, 16):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do transform_block_layout:

        .. code-block:: python

            sch = tir.Schedule(before_transform_block_layout)
            sch.transform_block_layout(sch.get_block("B"), lambda i, j: (i * 16 + j,))
            print(sch.mod["main"].script())

        After applying transform_block_layout, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_transform_block_layout(
                A: T.Buffer[(16, 16), "float32"],
                B: T.Buffer[(16, 16), "float32"]
            ) -> None:
                for i in range(256):
                    with T.block("B"):
                        vi, = T.axis.remap("S", [i])
                        B[vi // 16, vi % 16] = A[vi // 16, vi % 16] * 2.0
        """
        block = self._normalize_block_arg(block)
        if callable(index_map):
            index_map = IndexMap.from_func(index_map)
        _ffi_api.ScheduleTransformBlockLayout(  # type: ignore # pylint: disable=no-member
            self, block, index_map
        )

    @type_checked
    def set_axis_separator(
        self,
        block: Union[BlockRV, str],
        buffer: Union[Tuple[str, int], str, Buffer],
        axis_separators: Optional[List[int]],
    ) -> None:
        """Set the axis separator of a buffer, where the buffer is specified by a block and a read
        or write index.

        Parameters
        ----------
        block : Union[BlockRV, str]

            The block that accesses the target buffer.  If a string,
            this must uniquely identify a block.

        buffer: Union[Tuple[str,int], Buffer, str]

            The buffer to be transformed, or a specification of how to
            identify the buffer to be transformed.

            If `buffer` if a tuple of ``(str,int)``, the first item
            should be either "read" or "write", and the second item is
            an index into the block's read or write regions.

            If `buffer` is a string, it is the name of the buffer,
            which must exist within the reads/writes of the block.  In
            addition, the reads/writes of the block may not contain
            more than one buffer with this name.

            If `buffer` is a Buffer object, it must exist within the
            reads/writes of the block.

        axis_separators : Optional[List[int]]

            The axis separators.

        Examples
        --------

        Before set_axis_separator, in TensorIR, the IR is:

        .. code-block:: python

            @T.prim_func
            def before_set_axis_separator(
                A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]
            ) -> None:
                B = T.alloc_buffer((128, 128), dtype="float32")

                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do set_axis_separator:

        .. code-block:: python

            sch = tir.Schedule(before_set_axis_separator)
            sch.set_axis_separators(sch.get_block("B"), buffer_index=0, buffer_index_type="write",
                                    axis_separators=[1])
            print(sch.mod["main"].script())

        After applying set_axis_separator, the IR becomes:

        .. code-block:: python

            @T.prim_func
            def after_set_axis_separators(
                A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]
            ) -> None:
                B = T.alloc_buffer([128, 128], dtype="float32", axis_separators=[1])

                for i, j in T.grid(128, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * T.float32(2)
                for i, j in T.grid(128, 128):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + T.float32(1)
        """
        axis_separators = axis_separators or []

        block = self._normalize_block_arg(block)
        buffer_index_type, buffer_index, _ = self._normalize_buffer_arg(block, buffer)

        buffer_index_type_enum = 0 if buffer_index_type == "read" else 1
        _ffi_api.ScheduleSetAxisSeparator(  # type: ignore # pylint: disable=no-member
            self, block, buffer_index, buffer_index_type_enum, axis_separators
        )

    ########## Schedule: Misc ##########

    @type_checked
    def enter_postproc(self) -> None:
        """A no-op that marks the start of postprocessing phase of scheduling"""
        _ffi_api.ScheduleEnterPostproc(self)  # type: ignore # pylint: disable=no-member
