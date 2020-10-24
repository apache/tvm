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
"""Helper functions in TVM Script Parser"""

import inspect


def get_param_list(func):
    """Get the parameter list from definition of function"""
    full_arg_spec = inspect.getfullargspec(func)

    args, defaults = full_arg_spec.args, full_arg_spec.defaults

    if defaults is None:
        defaults = tuple()

    if full_arg_spec.varkw is not None:
        raise RuntimeError(
            "TVM Script register error : variable keyword argument is not supported now"
        )
    if not len(full_arg_spec.kwonlyargs) == 0:
        raise RuntimeError("TVM Script register error : keyword only argument is not supported now")

    pos_only = list()
    for arg in args[: len(args) - len(defaults)]:
        pos_only.append(arg)
    kwargs = list()
    for default, arg in zip(defaults, args[len(args) - len(defaults) :]):
        kwargs.append((arg, default))

    return pos_only, kwargs, full_arg_spec.varargs
