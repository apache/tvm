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
    """
    cache_by_key = {}
    def __init__(self, key):
        cache_dir = ".pkl_memoize_py{0}".format(sys.version_info[0])
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.path = os.path.join(cache_dir, key)
        if os.path.exists(self.path):
            try:
                self.cache = pickle.load(open(self.path, "rb"))
            except Exception:
                self.cache = {}
        else:
            self.cache = {}
        self.dirty = False

    def save(self):
        if self.dirty:
            print("Save memoize result to %s" % self.path)
            with open(self.path, "wb") as out_file:
                pickle.dump(self.cache, out_file, pickle.HIGHEST_PROTOCOL)

@atexit.register
def _atexit():
    """Save handler."""
    for value in Cache.cache_by_key.values():
        value.save()


def memoize(key):
    """Memoize the result of function and reuse multiple times.

    Parameters
    ----------
    key: str
        The unique key to the file

    Returns
    -------
    fmemoize : function
        The decorator function to perform memoization.
    """
    def _register(f):
        """Registration function"""
        allow_types = (string_types, int, float)
        fkey = key + "." + f.__name__ + ".pkl"
        if fkey not in Cache.cache_by_key:
            Cache.cache_by_key[fkey] = Cache(fkey)
        cache = Cache.cache_by_key[fkey]
        cargs = tuple(x.cell_contents for x in f.__closure__)
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
                print("Use memoize {0}{1}".format(fkey, key))
                return cache.cache[key]
            res = func(*args)
            cache.cache[key] = res
            cache.dirty = True
            return res

        return decorate(f, _memoized_f)

    return _register
