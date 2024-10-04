from typing import Any, List, Dict, Optional, Tuple, Union
from math import prod

class Node:
    def __init__(
            self, 
            node_id: int, 
            name: str, 
            sync_type: str, 
            hw_usage: str, 
            time: float, 
            input_buffer: List[str],
            output_buffer: str
        ):
        self.id = node_id
        self.name = name
        # self._out_edges = []
        # self._in_edges = []
        # self._shapes = []
        # self._dtypes = []
        # self._tag = {}
        self.shape = [16, 16]
        self.dtype = "float16"
        self.sync_type = sync_type
        self.hw_usage = hw_usage
        self.time = time
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        assert sync_type in ['sync', 'async'], 'invaild sync_type!'

    @property
    def dsize(self) -> int:
        assert self.dtype in ['float16', 'float32'], 'invaild dtype!'
        dtype_size = {'float16': 2, 'float32': 4}
        return dtype_size[self.dtype] * prod(self.shape)

    def __lt__(self, other: 'Node') -> bool:
        return self.id < other.id
    
    def __repr__(self) -> str:
        # return "<Node, " + self.name + ">"
        return self.name


import networkx as nx
from graph import *
import json

def load_model(fname: str) -> List[Node]:
    graph = nx.DiGraph()
    with open(fname) as f:
        a = json.load(f)
    node_map = {item[0] : None for item in a}
    for node_id, name, sync_type, hw_usage, input_buffer, output_buffer, inputs, time in a:
        input_list = []
        for src_node, src_id in inputs:
            if src_node not in node_map:
                input_list.append(None)
            else:
                assert node_map[src_node] is not None, "Detected ring in topo order {}->{} !".format(src_node, node_id)
                input_list.append([node_map[src_node], src_id])
        node = Node(node_id, name, sync_type, hw_usage, time, input_buffer, output_buffer)
        for src_node, _ in inputs:
            assert node_map[src_node] is not None
            graph.add_edge(node_map[src_node], node, weight=0, type="data")
        node_map[node_id] = node
    return graph