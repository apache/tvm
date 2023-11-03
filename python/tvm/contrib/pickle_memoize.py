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
"""Memoize result of function via pickle, used for cache testcases."""
# pylint: disable=broad-except,superfluous-parens
import os
import sys
import atexit
from decorator import decorate
from .._ffi.base import string_types

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Cache(object):
    """A cache object for result cache.

    Parameters
    ----------
    key: str
       The file key to the function
    save_at_exit: bool
        Whether save the cache to file when the program exits
    """

    cache_by_key = {}

    def __init__(self, key, save_at_exit):
        cache_dir = f".pkl_memoize_py{sys.version_info[0]}"
        try:
            os.mkdir(cache_dir)
        except FileExistsError:
            pass
        else:
            self.cache = {}
        self.path = os.path.join(cache_dir, key)
        if os.path.exists(self.path):
            try:
                self.cache = pickle.load(open(self.path, "rb"))
            except Exception:
                self.cache = {}
        else:
            self.cache = {}
        self.dirty = False
        self.save_at_exit = save_at_exit

    def save(self):
        if self.dirty:
            print(f"Save memoize result to {self.path}")
            with open(self.path, "wb") as out_file:
                pickle.dump(self.cache, out_file, pickle.HIGHEST_PROTOCOL)


@atexit.register
def _atexit():
    """Save handler."""
    for value in Cache.cache_by_key.values():
        if value.save_at_exit:
            value.save()


def memoize(key, save_at_exit=False):
    """Memoize the result of function and reuse multiple times.

    Parameters
    ----------
    key: str
        The unique key to the file
    save_at_exit: bool
        Whether save the cache to file when the program exits

    Returns
    -------
    fmemoize : function
        The decorator function to perform memoization.
    """

    def _register(f):
        """Registration function"""
        allow_types = (string_types, int, float, tuple)
        fkey = key + "." + f.__name__ + ".pkl"
        if fkey not in Cache.cache_by_key:
            Cache.cache_by_key[fkey] = Cache(fkey, save_at_exit)
        cache = Cache.cache_by_key[fkey]
        cargs = tuple(x.cell_contents for x in f.__closure__) if f.__closure__ else ()
        cargs = (len(cargs),) + cargs

        def _memoized_f(func, *args, **kwargs):
            assert not kwargs, "Only allow positional call"
            key = cargs + args
            for arg in key:
                if isinstance(arg, tuple):
                    for x in arg:
                        assert isinstance(x, allow_types)
                else:
                    assert isinstance(arg, allow_types)
            if key in cache.cache:
                return cache.cache[key]
            res = func(*args)
            cache.cache[key] = res
            cache.dirty = True
            return res

        return decorate(f, _memoized_f)

    return _register
