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
"""Relax Tuning Pass API primitives"""

from typing import Callable, Union, Dict, List, Optional, Sequence
import logging
import tvm
from tvm.runtime import Object
from tvm.ir.module import IRModule
from tvm.relax import Expr
from tvm.tir.schedule.trace import JSON_TYPE, _json_from_tvm
from tvm._ffi import register_object
from . import _ffi_api

logger = logging.getLogger("TuningAPI")  # pylint: disable=invalid-name


@register_object("relax.tuning_api.Choice")
class Choice(Object):
    """
    A TVM object Choice that maintains a set of transformation and constraint function keys.
    Corresponding functions should be registered as PackedFunc with these keys.
    Transformation function will be applied when constraint function returns true.
    Parameters
    ----------
    transform_func_key : Optional[str]
        Key for transformation function.
    transform_func_args : Optional[List]
        Arguments for transformation function.
    constr_func_key : Optional[str]
        Key for constraint function.
    constr_func_args : Optional[List]
        Arguments for constraint function.

    Examples
    --------
    The following code block defines a Choice.

    .. code-block:: python
        @tvm.register_func("relax.tuning_api.test.transform_func")
        def apply(mod):
            return relax.tuning_api.FoldConstant()(mod)
        @tvm.register_func("relax.tuning_api.test.constr_func")
        def constr(mod):
            return len(mod.functions) == 3
        # Define a choice to apply constant folding only when IRModule has three functions.
        choice = Choice(
            transform_func_key = "relax.tuning_api.test.transform_func",
            constr_func_key = "relax.tuning_api.test.constr_func"
            )
    """

    def __init__(
        self,
        transform_func_key: Optional[str] = None,
        transform_func_args: Optional[List] = None,
        constr_func_key: Optional[str] = None,
        constr_func_args: Optional[List] = None,
    ):
        """Constructor
        Parameters
        ----------
        transform_func_key : Optional[str]
            Key for transformation function.

        f_tramsform_args: Optional[List]
            Arguments for transformation function.

        constr_func_key : Optional[str]
            Key for constraint function.

        constr_func_args: Optional[List]
            Arguments for constraint function.
        """

        if transform_func_key is None:
            transform_func_key = "relax.tuning_api.Choice.default_transform_func"

        if transform_func_args is None:
            transform_func_args = []

        if constr_func_key is None:
            constr_func_key = "relax.tuning_api.Choice.default_constr_func"

        if constr_func_args is None:
            constr_func_args = []

        self.__init_handle_by_constructor__(
            _ffi_api.Choice,  # type: ignore
            transform_func_key,
            transform_func_args,
            constr_func_key,
            constr_func_args,  # type: ignore # pylint: disable=no-member
        )

    def get_transform_func(self) -> Callable:
        """Getter for transform_func
        Returns
        -------
        ret: Callable
           registered transformation function
        """
        return _ffi_api.ChoiceGetTransformFunc(self)  # type: ignore

    def get_constr_func(self) -> Callable:
        """Getter for constr_func
        Returns
        -------
        ret: Callable
           registered constraint function
        """
        return _ffi_api.ChoiceGetConstrFunc(self)  # type: ignore

    def apply_transform_func(self, mod: IRModule) -> IRModule:
        """Perform transform_func with its arguments
        Returns
        -------
        ret: IRModule
           Transformed IRModule
        """
        return _ffi_api.ChoiceApplyTransformFunc(self, mod)  # type: ignore

    def check_constr(self, mod: IRModule) -> bool:
        """Perform constr_func with its arguments
        Returns
        -------
        ret: bool
           Returns whether the IRModule satisfies the constraint or not
        """
        return _ffi_api.ChoiceCheckConstr(self, mod)  # type: ignore

    def as_json(self) -> JSON_TYPE:
        """Serialize the trace as a JSON-style object
        Returns
        -------
        json: JSON_TYPE
            The JSON-style object
        """
        return _ffi_api.ChoiceAsJSON(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: JSON_TYPE) -> "Choice":
        """Create Choice from JSON obj

        Parameters
        ----------
        json_obj: JSON_TYPE
            Choice serialized with JSON

        Return
        ----------
        choice: Choice
            Deserialized choice
        """
        return _ffi_api.ChoiceFromJSON(json_obj)  # type: ignore

    def deepcopy(self):
        return Choice.from_json(self.as_json())


@register_object("relax.tuning_api.Knob")
class Knob(Object):
    """
    A TVM object Knob that maintains a set of valid Choices.
    By using Knobs, a tuning pass can generate candidates and define the search space.
    Parameters
    ----------
    name : str
        Name of the knob.

    choices: Union[List[Choice], Dict[str, Choice]]
        A list of valid choices

    Examples
    --------
    The following code block defines a Knob.

    .. code-block:: python
        @tvm.register_func("relax.tuning_api.test.transform_func")
        def apply(mod):
            return relax.tuning_api.FoldConstant()(mod)
        choices = {"apply": Choice("relax.tuning_api.test.transform_func"), "noapply": Choice()}
        # A knob manages a set of its valid choices
        knob = Knob("MockTuningKnob", choices)
    """

    def __init__(self, name: str, choices: Union[List[Choice], Dict[str, Choice]]):
        """Constructor."""
        if isinstance(choices, list):
            choices = {str(idx): val for idx, val in enumerate(choices)}

        self.__init_handle_by_constructor__(
            _ffi_api.Knob, name, choices  # type: ignore # pylint: disable=no-member
        )

    def verify(self, decision: Union[str, int]) -> bool:
        """Verify if the decision is valid."""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.KnobIsValidDecision(self, decision)  # type: ignore

    def apply(self, mod: IRModule, decision: Union[str, int]) -> IRModule:
        """Get choice if a decision is valid."""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.KnobApply(self, mod, decision)  # type: ignore

    def as_json(self) -> JSON_TYPE:
        """Serialize the trace as a JSON-style object
        Returns
        -------
        json: JSON_TYPE
            The JSON-style object
        """
        return _ffi_api.KnobAsJSON(self)  # type: ignore

    @staticmethod
    def from_json(json_obj: JSON_TYPE) -> "Knob":
        """Create Knob from JSON obj

        Parameters
        ----------
        json_obj: JSON_TYPE
            Knob serialized with JSON

        Return
        ----------
        knob: Knob
            Deserialized knob
        """
        return _ffi_api.KnobFromJSON(json_obj)  # type: ignore

    def __str__(self) -> str:
        msg = f"{self.name} (# of choices: {len(self.choices)})\n"
        for name, choice in self.choices.items():
            msg += f"  - {name}: {choice}\n"
        return msg

    def deepcopy(self):
        return Knob.from_json(self.as_json())


@register_object("relax.tuning_api.Trace")
class Trace(Object):
    """
    A TVM object Trace logs the history of transformations (decisions).
    Parameters
    ----------
    in_mod : IRModule
        Input IRModule.
    knobs: Optional[List[Knob]]
        A list of knobs applied in the trace.
    decisions: Optional[Sequence[Union[str, int]]]
        A list of decisions made for each knob

    Examples
    --------
    The following code block defines a Trace.

    .. code-block:: python

        trace = Trace(mod, [knob1, knob2, knob3], ["c1", "c0", "c3"])
        assert trace.size == 3 # Length of history.
        # 'out' contains IRModule that applies transformations in the trace.
        out: IRModule = trace.add(knob4, "c2")
        assert trace.size == 4 # Length of history.
        trace.set_perf(0.03) # Set the performance number of the trace.
    """

    def __init__(
        self,
        in_mod: IRModule,
        knobs: Optional[List[Knob]] = None,
        decisions: Optional[Sequence[Union[str, int]]] = None,
    ):
        """Constructor."""
        knobs = knobs if knobs else list()
        decisions = (
            [str(v) if isinstance(v, int) else v for v in decisions] if decisions else list()
        )
        self.__init_handle_by_constructor__(
            _ffi_api.Trace, in_mod, knobs, decisions  # type: ignore # pylint: disable=no-member
        )

    def verify(self) -> bool:
        """Verify if current history is valid."""
        return _ffi_api.TraceVerify()  # type: ignore

    def add(self, knob: Knob, decision: Union[str, int]) -> IRModule:
        """Add & Apply new decision (with knob)."""
        if isinstance(decision, int):
            decision = str(decision)
        return _ffi_api.TraceAdd(self, knob, decision)  # type: ignore

    def set_perf(self, perf: float) -> None:
        """Set performance number for the trace."""
        return _ffi_api.TraceSetPerf(self, perf)  # type: ignore

    def set_out_mod(self, mod: IRModule) -> None:
        """Set out_mod for the trace."""
        return _ffi_api.TraceSetOutMod(self, mod)  # type: ignore

    def as_json(self, include_irmod: bool = True) -> JSON_TYPE:
        """Serialize the trace as a JSON-style object.
        Parameters
        ----------
        include_irmod: bool
            Decides whether to serialize in_mod as well.

        Returns
        -------
        json: JSON_TYPE
            The JSON-style object.
        """
        obj = _ffi_api.TraceAsJSON(self, include_irmod)  # type: ignore
        return _json_from_tvm(obj)

    @staticmethod
    def from_json(json_obj: JSON_TYPE) -> "Trace":
        """Create Trace from JSON obj.

        Parameters
        ----------
        json_obj: JSON_TYPE
            Trace serialized with JSON.

        Return
        ----------
        trace: Trace
            Deserialized trace.
        """
        return _ffi_api.TraceFromJSON(json_obj)  # type: ignore

    def __str__(self) -> str:
        n = len(self.knobs)
        msg = f"Trace length: {n}\n"
        for idx in range(n):
            msg += f"[{idx+1}] {self.knobs[idx].name}: {self.decisions[idx]}\n"
        return msg

    def deepcopy(self) -> "Trace":
        new_in_mod = deepcopy_irmodule(self.in_mod)
        new_knobs = [knob.deepcopy() for knob in self.knobs]
        new_decisions = [str(decision) for decision in self.decisions]
        new_trace = Trace(new_in_mod, new_knobs, new_decisions)
        new_out_mod = deepcopy_irmodule(self.out_mod)
        new_trace.set_out_mod(new_out_mod)
        return new_trace


def get_trace(in_: Union[Trace, IRModule, Expr]) -> Trace:
    """
    Getter for a trace wrapper.

    Parameters
    ----------
    in_: Union[Trace, IRModule, Expr]
        Input entity
    Return
    ----------
    wrapped: Trace
        Traced entity
    """
    if isinstance(in_, Trace):
        return in_
    if isinstance(in_, IRModule):
        return Trace(in_)
    if isinstance(in_, Expr):  # type: ignore
        return Trace(tvm.IRModule.from_expr(in_))

    raise Exception(f"Invalid input type for trace: {type(in_)}")


@tvm.register_func("relax.tuning_api.deepcopy_irmodule")
def deepcopy_irmodule(mod: IRModule) -> IRModule:
    """
    Deepcopy for an IRModule.
    Parameters
    ----------
    mod: IRModule
        input IRModule
    Return
    ----------
    copied_mod: IRModule
        deep-copied IRModule
    """
    func_save_json = tvm.get_global_func("node.SaveJSON")
    func_load_json = tvm.get_global_func("node.LoadJSON")
    new_mod = None
    # Handle external modules separately if exist
    # TODO(tvm-team):
    #   Serialization of IRModule with external mods is tricky.
    #   (1) External mod is runtime module.
    #   (2) Currently, `export_library` does not support serialization of
    #       runtime module without the host module
    #   Therefore, we simply pass around the compiled external modules without copy for now.
    #   Revisit later when we have a better solution.
    if mod.attrs and "external_mods" in mod.attrs:
        tmp_mod = mod.without_attr("external_mods")
        new_mod = func_load_json(func_save_json(tmp_mod))
        new_mod = new_mod.with_attr("external_mods", mod.attrs["external_mods"])
    else:
        new_mod = func_load_json(func_save_json(mod))

    return new_mod
