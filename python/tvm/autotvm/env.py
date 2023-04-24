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
# pylint: disable=global-variable-not-assigned
"""Global configuration/variable scope for autotvm"""


class AutotvmGlobalScope(object):
    """The global autotvm scope."""

    current = None

    def __init__(self):
        self._old = AutotvmGlobalScope.current
        AutotvmGlobalScope.current = self

        self.in_tuning = False
        self.silent = False

    def deep_copy(self, global_scope):
        """Deep copy from another instance of AutotvmGlobalScope."""
        self._old = AutotvmGlobalScope.current

        self.in_tuning = global_scope.in_tuning
        self.silent = global_scope.silent


GLOBAL_SCOPE = AutotvmGlobalScope()


def reset_global_scope(global_scope):
    """Reset global autotvm state. This is needed to initialize PopenPool workers."""
    global GLOBAL_SCOPE
    GLOBAL_SCOPE.deep_copy(global_scope)
    AutotvmGlobalScope.current = global_scope
