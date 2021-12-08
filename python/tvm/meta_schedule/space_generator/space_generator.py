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
from typing import TYPE_CHECKING, List

from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from .. import _ffi_api
from ..utils import check_override

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.SpaceGenerator")
class SpaceGenerator(Object):
    """The abstract design space generator interface."""

    def initialize_with_tune_context(
        self,
        tune_context: "TuneContext",
    ) -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initializing the design space generator.
        """
        _ffi_api.SpaceGeneratorInitializeWithTuneContext(  # type: ignore # pylint: disable=no-member
            self, tune_context
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


@register_object("meta_schedule.PySpaceGenerator")
class PySpaceGenerator(SpaceGenerator):
    """An abstract design space generator with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        @check_override(self.__class__, SpaceGenerator)
        def f_initialize_with_tune_context(tune_context: "TuneContext") -> None:
            self.initialize_with_tune_context(tune_context)

        @check_override(self.__class__, SpaceGenerator)
        def f_generate_design_space(mod: IRModule) -> List[Schedule]:
            return self.generate_design_space(mod)

        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorPySpaceGenerator,  # type: ignore # pylint: disable=no-member
            f_initialize_with_tune_context,
            f_generate_design_space,
        )
