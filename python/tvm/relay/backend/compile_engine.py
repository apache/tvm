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
"""Backend code generation engine."""
from __future__ import absolute_import

from ..base import register_relay_node, NodeBase
from ... import target as _target
from .. import expr as _expr
from . import _backend

@register_relay_node
class CachedFunc(NodeBase):
    """Low-level tensor function to back a relay primitive function.
    """


@register_relay_node
class CCacheKey(NodeBase):
    """Key in the CompileEngine.

    Parameters
    ----------
    source_func : tvm.relay.Function
        The source function.

    target : tvm.Target
        The target we want to run the function on.
    """
    def __init__(self, source_func, target):
        self.__init_handle_by_constructor__(
            _backend._make_CCacheKey, source_func, target)


@register_relay_node
class CCacheValue(NodeBase):
    """Value in the CompileEngine, including usage statistics.
    """


def _get_cache_key(source_func, target):
    if isinstance(source_func, _expr.Function):
        if isinstance(target, str):
            target = _target.create(target)
            if not target:
                raise ValueError("Need target when source_func is a Function")
        return CCacheKey(source_func, target)
    if not isinstance(source_func, CCacheKey):
        raise TypeError("Expect source_func to be CCacheKey")
    return source_func


@register_relay_node
class CompileEngine(NodeBase):
    """CompileEngine to get lowered code.
    """
    def __init__(self):
        raise RuntimeError("Cannot construct a CompileEngine")

    def lower(self, source_func, target=None):
        """Lower a source_func to a CachedFunc.

        Parameters
        ----------
        source_func : Union[tvm.relay.Function, CCacheKey]
            The source relay function.

        target : tvm.Target
            The target platform.

        Returns
        -------
        cached_func: CachedFunc
            The result of lowering.
        """
        # pylint: disable=broad-except
        try:
            key = _get_cache_key(source_func, target)
            return _backend._CompileEngineLower(self, key)
        except Exception:
            import traceback
            msg = traceback.format_exc()
            msg += "Error during compile func\n"
            msg += "--------------------------\n"
            msg += source_func.astext(show_meta_data=False)
            msg += "--------------------------\n"
            raise RuntimeError(msg)

    def jit(self, source_func, target=None):
        """JIT a source_func to a tvm.Function.

        Parameters
        ----------
        source_func : Union[tvm.relay.Function, CCacheKey]
            The source relay function.

        target : tvm.Target
            The target platform.

        Returns
        -------
        cached_func: CachedFunc
            The result of lowering.
        """
        key = _get_cache_key(source_func, target)
        return _backend._CompileEngineJIT(self, key)

    def clear(self):
        """clear the existing cached functions"""
        _backend._CompileEngineClear(self)

    def items(self):
        """List items in the cache.

        Returns
        -------
        item_list : List[Tuple[CCacheKey, CCacheValue]]
            The list of items.
        """
        res = _backend._CompileEngineListItems(self)
        assert len(res) % 2 == 0
        return [(res[2*i], res[2*i+1]) for i in range(len(res) // 2)]

    def dump(self):
        """Return a string representation of engine dump.

        Returns
        -------
        dump : str
            The dumped string representation
        """
        items = self.items()
        res = "====================================\n"
        res += "CompilerEngine dump, %d items cached\n" % len(items)
        for k, v in items:
            res += "------------------------------------\n"
            res += "target={}\n".format(k.target)
            res += "use_count={}\n".format(v.use_count)
            res += "func_name={}\n".format(v.cached_func.func_name)
            res += k.source_func.astext() + "\n"
        res += "===================================\n"
        return res


def get():
    """Get the global compile engine.

    Returns
    -------
    engine : tvm.relay.backend.CompileEngine
        The compile engine.
    """
    return _backend._CompileEngineGlobal()
