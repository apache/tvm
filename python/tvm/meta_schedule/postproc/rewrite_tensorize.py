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
"""A postprocessor that tensorize related components."""

from tvm._ffi.registry import register_object
from .. import _ffi_api
from .postproc import Postproc


@register_object("meta_schedule.RewriteTensorize")
class RewriteTensorize(Postproc):
    """A postprocessor that applies tensorization to annotated blocks.

    Parameters
    ----------
    vectorize_init_loop : bool
       Whether or not vectorize the initialization loop produced by DecomposeReduction
    """

    def __init__(self, vectorize_init_loop=False) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.PostprocRewriteTensorize,  # type: ignore # pylint: disable=no-member
            vectorize_init_loop,
        )
