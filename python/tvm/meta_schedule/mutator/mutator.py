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
"""Meta Schedule Mutator."""
from typing import Optional, TYPE_CHECKING

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Trace

from ..utils import _get_hex_address
from .. import _ffi_api

if TYPE_CHECKING:
    from ..tune_context import TuneContext


class Mutator(Object):
    """Mutator is designed to mutate the trace to explore the design space."""

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        """Initialize the mutator with a tune context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initializing the design space generator.
        """
        _ffi_api.MutatorInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, tune_context
        )

    def apply(self, trace: Trace) -> Optional[Trace]:
        """Apply the mutator function to the given trace.

        Parameters
        ----------
        trace : Trace
            The given trace for mutation.

        Returns
        -------
        trace : Optional[Trace]
            None if mutator failed, otherwise return the mutated trace.
        """
        return _ffi_api.MutatorApply(self, trace)


@register_object("meta_schedule.PyMutator")
class PyMutator(Mutator):
    """An abstract mutator with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        def f_initialize_with_tune_context(tune_context: "TuneContext") -> None:
            self.initialize_with_tune_context(tune_context)

        def f_apply(trace: Trace) -> Optional[Trace]:
            return self.apply(trace)

        def f_as_string() -> str:
            return str(self)

        self.__init_handle_by_constructor__(
            _ffi_api.MutatorPyMutator,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_apply,
            f_as_string,
        )

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        raise NotImplementedError

    def apply(self, trace: Trace) -> Optional[Trace]:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"PyMutator({_get_hex_address(self.handle)})"
