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
"""Utilities for thrust"""
import logging

from tvm._ffi import get_global_func


def maybe_warn(target, func_name):
    if get_global_func(func_name, allow_missing=True) and not "thrust" in target.libs:
        logging.warning("TVM is built with thrust but thrust is not used.")
    if "thrust" in target.libs and get_global_func(func_name, allow_missing=True) is None:
        logging.warning("thrust is requested but TVM is not built with thrust.")


def can_use_thrust(target, func_name):
    maybe_warn(target, func_name)
    return (
        target.kind.name in ["cuda", "nvptx"]
        and "thrust" in target.libs
        and get_global_func(func_name, allow_missing=True)
    )


def can_use_rocthrust(target, func_name):
    maybe_warn(target, func_name)
    return (
        target.kind.name == "rocm"
        and "thrust" in target.libs
        and get_global_func(func_name, allow_missing=True)
    )
