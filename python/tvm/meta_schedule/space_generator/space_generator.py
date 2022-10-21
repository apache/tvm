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
Meta Schedule design space generators that generates design
space for generation of measure candidates.
"""
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

# isort: off
from typing_extensions import Literal

# isort: on
from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from .. import _ffi_api

if TYPE_CHECKING:
    from ..mutator import Mutator
    from ..postproc import Postproc
    from ..schedule_rule import ScheduleRule
    from ..tune_context import TuneContext


@register_object("meta_schedule.SpaceGenerator")
class SpaceGenerator(Object):
    """The abstract design space generator interface."""

    ScheduleFnType = Union[
        Callable[[Schedule], None],  # No output
        Callable[[Schedule], Schedule],  # Single output
        Callable[[Schedule], List[Schedule]],  # Multiple outputs
    ]

    SpaceGeneratorType = Union[
        "SpaceGenerator",
        ScheduleFnType,
        Literal["post-order-apply", "union"],
    ]

    sch_rules: Optional[List["ScheduleRule"]]
    postprocs: Optional[List["Postproc"]]
    mutator_probs: Optional[Dict["Mutator", float]]

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the design space generator.
        """
        _ffi_api.SpaceGeneratorInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, context
        )

    def generate_design_space(self, mod: IRModule) -> List[Schedule]:
        """Generate design spaces given a module.

        Parameters
        ----------
        mod : IRModule
            The module used for design space generation.

        Returns
        -------
        design_spaces : List[Schedule]
            The generated design spaces, i.e., schedules.
        """
        return _ffi_api.SpaceGeneratorGenerateDesignSpace(self, mod)  # type: ignore # pylint: disable=no-member

    def clone(self) -> "SpaceGenerator":
        """Clone the design space generator.

        Returns
        -------
        cloned_sg : SpaceGenerator
            The cloned design space generator.
        """
        return _ffi_api.SpaceGeneratorClone(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(  # pylint: disable=keyword-arg-before-vararg
        kind: Union[
            Literal["post-order-apply", "union"],
            ScheduleFnType,
        ] = "post-order-apply",
        *args,
        **kwargs,
    ) -> "SpaceGenerator":
        """Create a design space generator."""
        from . import (  # pylint: disable=import-outside-toplevel
            PostOrderApply,
            ScheduleFn,
            SpaceGeneratorUnion,
        )

        if callable(kind):

            def create_schedule_fn(
                func,
                sch_rules=[],
                postprocs=[],
                mutator_probs={},
            ):  # pylint: disable=dangerous-default-value
                return ScheduleFn(func, sch_rules, postprocs, mutator_probs)

            return create_schedule_fn(kind, *args, **kwargs)  # type: ignore
        if kind == "post-order-apply":
            return PostOrderApply(*args, **kwargs)
        if kind == "union":
            return SpaceGeneratorUnion(*args, **kwargs)
        raise ValueError(f"Unknown SpaceGenerator: {kind}")


ScheduleFnType = SpaceGenerator.ScheduleFnType
ScheduleRuleType = Union[
    List["ScheduleRule"],
    Literal["llvm", "cuda", "cuda-tensorcore", "hexagon", "from-target"],
]
PostprocType = Union[
    List["Postproc"],
    Literal["llvm", "cuda", "cuda-tensorcore", "hexagon", "from-target"],
]
MutatorProbType = Union[
    Dict["Mutator", float],
    Literal["llvm", "cuda", "cuda-tensorcore", "hexagon", "from-target"],
]
create = SpaceGenerator.create  # pylint: disable=invalid-name


def _normalize_rules(
    sch_rules: ScheduleRuleType,
    postprocs: PostprocType,
    mutator_probs: MutatorProbType,
) -> Tuple[
    Optional[List["ScheduleRule"]],
    Optional[List["Postproc"]],
    Optional[Dict["Mutator", float]],
]:
    # pylint: disable=import-outside-toplevel
    from ..mutator import Mutator
    from ..postproc import Postproc
    from ..schedule_rule import ScheduleRule

    # pylint: enable=import-outside-toplevel
    assert sch_rules is not None
    assert postprocs is not None
    assert mutator_probs is not None

    if isinstance(sch_rules, str):
        if sch_rules == "from-target":
            sch_rules = None
        else:
            sch_rules = ScheduleRule.create(sch_rules)
    if isinstance(postprocs, str):
        if postprocs == "from-target":
            postprocs = None
        else:
            postprocs = Postproc.create(postprocs)
    if isinstance(mutator_probs, str):
        if mutator_probs == "from-target":
            mutator_probs = None
        else:
            mutator_probs = Mutator.create(mutator_probs)
    return sch_rules, postprocs, mutator_probs  # type: ignore


@register_object("meta_schedule.PySpaceGenerator")
class _PySpaceGenerator(SpaceGenerator):
    """
    A TVM object space generator to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PySpaceGenerator
    """

    def __init__(
        self,
        sch_rules: ScheduleRuleType = "from-target",
        postprocs: PostprocType = "from-target",
        mutator_probs: MutatorProbType = "from-target",
        f_initialize_with_tune_context: Optional[Callable] = None,
        f_generate_design_space: Optional[Callable] = None,
        f_clone: Optional[Callable] = None,
    ):
        """Constructor."""
        sch_rules, postprocs, mutator_probs = _normalize_rules(sch_rules, postprocs, mutator_probs)

        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorPySpaceGenerator,  # type: ignore # pylint: disable=no-member
            sch_rules,
            postprocs,
            mutator_probs,
            f_initialize_with_tune_context,
            f_generate_design_space,
            f_clone,
        )


class PySpaceGenerator:
    """
    An abstract space generator with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PySpaceGenerator,
        "fields": ["sch_rules", "postprocs", "mutator_probs"],
        "methods": ["_initialize_with_tune_context", "generate_design_space", "clone"],
    }

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initializing the design space generator.
        """
        raise NotImplementedError

    def generate_design_space(self, mod: IRModule) -> List[Schedule]:
        """Generate design spaces given a module.

        Parameters
        ----------
        mod : IRModule
            The module used for design space generation.

        Returns
        -------
        design_spaces : List[Schedule]
            The generated design spaces, i.e., schedules.
        """
        raise NotImplementedError

    def clone(self) -> SpaceGenerator:
        """Clone the design space generator.

        Returns
        -------
        cloned_sg : SpaceGenerator
            The cloned design space generator.
        """
        raise NotImplementedError
