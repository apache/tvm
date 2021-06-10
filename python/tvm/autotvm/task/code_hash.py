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
Decorator functions for hashing schedule code

code hashing is used to check the consistence of schedule code and the parameters loaded from log
"""
import functools
import inspect
import zlib

from tvm.te import schedule


def attach_code_hash(s):
    """Decorator for attaching a code hash to a schedule

    Parameters
    ----------
    s: Schedule
        tvm.te.schedule.Schedule to attach the hash to
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            raw_hash = zlib.crc32("".join(inspect.getsourcelines(func)[0]).encode())
            s.code_hash = hex(raw_hash)[2:]

        return wrapper

    return decorator


def attach_code_hash_to_arg(arg_idx=1):
    """Decorator for attaching a code hash to a schedule

    Parameters
    ----------
    arg_idx: int
        index of the argument (expected to be a Schedule) to attach the code
        hash to
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            assert isinstance(args[arg_idx], schedule.Schedule)
            raw_hash = zlib.crc32("".join(inspect.getsourcelines(func)[0]).encode())
            args[arg_idx].code_hash = hex(raw_hash)[2:]

        return wrapper

    return decorator
