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
# pylint: disable=invalid-name
"""Compiler engine interface to internal engine

You can get the engine singleton at ``nnvm.compiler.engine``
"""
import tvm

_list_cache_items = tvm.get_global_func("nnvm.compiler.ListCacheItems")
_clear_cache = tvm.get_global_func("nnvm.compiler.ClearCache")
_get_cache_item = tvm.get_global_func("nnvm.compiler.GetCacheItem")
_set_cache_item = tvm.get_global_func("nnvm.compiler.SetCacheItem")
_graph_key_get_graph = tvm.get_global_func("nnvm.compiler.GraphKeyGetGraph")
_make_graph_key = tvm.get_global_func("nnvm.compiler.MakeGraphKey")

@tvm.register_node
class GraphKey(tvm.node.NodeBase):
    """Key of a graph compilation context"""
    @property
    def graph(self):
        return _graph_key_get_graph(self)


@tvm.register_node
class GraphCacheEntry(tvm.node.NodeBase):
    """CacheEntry of compilation into a TVM Function"""


@tvm.register_node
class GraphFunc(tvm.node.NodeBase):
    """Compiled result of a graph into a TVM Function"""


class Engine(object):
    """Global singleton compilation engine.

    You can get the singleton at ``nnvm.compiler.engine``
    """
    def items(self):
        """List the available cache key value pairs.

        Returns
        -------
        item_list : list of (GraphKey, GraphCacheEntry)
            The existing cache items
        """
        res = _list_cache_items()
        assert len(res) % 2 == 0
        return [(res[2*i], res[2*i+1]) for i in range(len(res) // 2)]

    def clear_cache(self):
        """Clear the existing cached functions."""
        _clear_cache()

    def __setitem__(self, key, value):
        """Clear the existing cached functions."""
        if isinstance(value, GraphCacheEntry):
            _set_cache_item(key, value.graph_func)
        else:
            _set_cache_item(key, value)

    def __getitem__(self, key):
        """Clear the existing cached functions."""
        return _get_cache_item(key)

    def dump(self):
        """Return a string representation of engine dump

        Returns
        -------
        dump : str
            The dumped string representation
        """
        items = self.items()
        res = "====================================\n"
        res += "CompilerEngine dump, %d items cached\n" % len(items)
        for key, value in items:
            res += "------------------------------------\n"
            res += "target={}\n".format(key.target)
            res += "inputs={}\n".format(key.inputs)
            res += "use_count={}\n".format(value.use_count)
            res += "func_name={}\n".format(value.graph_func.func_name)
            res += key.graph.ir() + "\n"
        res += "===================================\n"
        return res

engine = Engine()


def graph_key(graph, inputs, target):
    """Construct a new graph key.

    Parameters
    ----------
    graph : Graph
        The computation graph structure

    inputs : list of Tensor(placeholder)
        The input requirement to the graph.

    target : str
        The target of compilation.
    """
    return _make_graph_key(graph, inputs, target)
