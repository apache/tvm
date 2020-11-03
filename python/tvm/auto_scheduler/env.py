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
"""The scope to store global environmental variables of the auto-scheduler"""


class AutoSchedulerGlobalScope(object):
    """The global scope to store environmental variables of the auot-scheduler"""

    def __init__(self):
        self.enable_relay_integration = False


GLOBAL_SCOPE = AutoSchedulerGlobalScope()


def is_relay_integration_enabled():
    """Return whether the relay integration is enabled

    Returns
    -------
    enabled: bool
        Whether the relay integration is enabled
    """
    return GLOBAL_SCOPE.enable_relay_integration


def enable_relay_integration(new_value=True):
    """Set the relay integration

    Parameters
    ---------
    new_value: bool = True
        The new setting of relay integration

    Returns
    -------
    old_value: bool
        The old setting.
    """
    old_value = GLOBAL_SCOPE.enable_relay_integration
    GLOBAL_SCOPE.enable_relay_integration = new_value
    return old_value
