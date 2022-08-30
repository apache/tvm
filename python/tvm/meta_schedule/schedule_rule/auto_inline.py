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
"""Auto-Inline. Rule that inlines spatial blocks if it satisfies some conditions"""
from typing import List, Optional

from tvm._ffi import register_object

from .. import _ffi_api
from .schedule_rule import ScheduleRule


@register_object("meta_schedule.AutoInline")
class AutoInline(ScheduleRule):
    """Rule that inlines spatial blocks if it satisfies some conditions

    Parameters
    ----------
    into_producer : bool
        If allows to inline a block into its producer
    into_consumer : bool
        If allows to inline a block into its consumer
    inline_const_tensor : bool
        Always inline constant tensors
    disallow_if_then_else : bool
        Always disallow if-then-else-like constructs
    require_injective : bool
        Always require the read-to-write mapping to be ordered
    require_ordered : bool
        Always require the read-to-write mapping to be injective
    disallow_op : Optional[List[str]]
        The operators that are disallowed in auto inline
    """

    def __init__(
        self,
        into_producer: bool,
        into_consumer: bool,
        inline_const_tensor: bool,
        disallow_if_then_else: bool,
        require_injective: bool,
        require_ordered: bool,
        disallow_op: Optional[List[str]] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleAutoInline,  # type: ignore # pylint: disable=no-member
            into_producer,
            into_consumer,
            inline_const_tensor,
            disallow_if_then_else,
            require_injective,
            require_ordered,
            disallow_op,
        )
