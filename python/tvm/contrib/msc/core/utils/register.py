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
"""tvm.contrib.msc.core.utils.register"""

from .namespace import MSCMap, MSCKey, MSCFramework


def register_func(name: str, func: callable, framework: str = MSCFramework.MSC):
    """Register a func for framework.

    Parameters
    ----------
    name: string
        The name for the func.
    func: callable
        The function to be registered.
    framework: string
        Should be from MSCFramework.
    """

    funcs = MSCMap.get(MSCKey.REGISTERED_FUNCS, {})
    if framework not in funcs:
        funcs[framework] = {}
    funcs[framework][name] = func
    MSCMap.set(MSCKey.REGISTERED_FUNCS, funcs)


def get_registered_func(name: str, framework: str = MSCFramework.MSC):
    """Get the registered func of framework.

    Parameters
    ----------
    name: string
        The name for the func.
    framework: string
        Should be from MSCFramework.

    Returns
    -------
    func: callable
        The registered function.
    """

    funcs = MSCMap.get(MSCKey.REGISTERED_FUNCS, {})
    if framework not in funcs:
        return None
    return funcs[framework].get(name)
