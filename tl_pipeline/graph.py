from typing import Any, List, Dict, Optional, Tuple, Union
import heapq

class Edge:
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int, w: int):
        self.w = w
        self.src_node = src_node
        self.dst_node = dst_node
        self.src_id = src_id
        self.dst_id = dst_id
    
    def __repr__(self) -> str:
        return "<Edge, " + self.src_node.name + "--(" + str(self.w) + ")-->" + self.dst_node.name + ">"

class DataDependencyEdge(Edge):
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int, w: int):
        super().__init__(src_node, dst_node, src_id, dst_id, w)

    def __repr__(self) -> str:
        return "<DataDependencyEdge, " + self.src_node.name + "--(" + str(self.w) + ")-->" + self.dst_node.name + ">"

class CondDependencyEdge(Edge):
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int, w: int):
        super().__init__(src_node, dst_node, src_id, dst_id, w)

    def __repr__(self) -> str:
        return "<CondDependencyEdge, " + self.src_node.name + "--(" + str(self.w) + ")-->" + self.dst_node.name + ">"

class Node:
    def __init__(self, node_id: int, inputs: List[Union[Tuple['Node', int], 'Node', None]], name: str, sync_type: str, hw_usage: List[str], buffer_layer: List[str]):
        self.id = node_id
        self.name = name
        self._out_edges = []
        self._in_edges = []
        self._shapes = []
        self._dtypes = []
        self._tag = {}
        self._sync_type = sync_type
        self.hw_usage = hw_usage
        self.buffer_layer = buffer_layer
        assert sync_type in ['sync', 'async'], 'invaild sync_type!'

        for i, node in enumerate(inputs):
            if node is None:
                inputs[i] = PlaceHolderNode()

        for dst_id, n in enumerate(inputs):
            if isinstance(n, Node):
                n = (n, 0)
            assert(len(n) == 2)
            src_node, src_id = n[0], n[1]
            edge = DataDependencyEdge(src_node, self, src_id, dst_id, 0)
            self._in_edges.append(edge)
            src_node._out_edges.append(edge)

    @property
    def inputs(self) -> List[Edge]:
        return self._in_edges

    @property
    def outputs(self) -> List[Edge]:
        return self._out_edges

    def set_inputs(self, i: int, edge: Edge):
        assert i < len(self._in_edges)
        self._in_edges[i] = edge

    def set_outputs(self, i: int, edge: Edge):
        assert i < len(self._out_edges)
        self._out_edges[i] = edge

    def get_shape(self, id: int = 0) -> List[int]:
        return self._shapes[id]

    def set_shape(self, shape: List[int], id=0, overwrite=False) -> None:
        if len(self._shapes) <= id:
            self._shapes.extend([None for _ in range(id - len(self._shapes) + 1)])
        elif self._shapes[id] is not None and not overwrite:
            assert self._shapes[id] == list(map(int, shape)), (self._shapes, list(map(int, shape)))
        self._shapes[id] = list(map(int, shape))

    def is_placeholder(self):
        return False

    def is_output(self):
        return False

    def add_tag(self, k: str, v: Any = True) -> None:
        self._tag[k] = v

    def get_tag(self, k: str) -> Any:
        if k not in self._tag:
            return None
        return self._tag[k]

    def num_outputs(self) -> int:
        if len(self.outputs) == 0:
            return 0
        return max([e.src_id for e in self.outputs]) + 1

    def get_ir(self) -> str:
        raise NotImplementedError()

    def __repr__(self) -> str:
        # return "<Node, " + self.name + ">"
        return self.name

class PlaceHolderNode(Node):
    def __init__(self, name=""):
        super().__init__([], "PlaceHolder " + name, "sync")

    def is_placeholder(self):
        return True

    def get_ir(self) -> str:
        return "placeholder"

class OutputNode(Node):
    def __init__(self, node, id=0):
        super().__init__([(node, id)], "Output ", "sync")
        # self.set_shape(node.get_shape(id))
        # self.set_dtype(node.get_dtype(id))

    def is_output(self):
        return True

    def get_ir(self) -> str:
        return "output"

class Graph:
    def __init__(self, nodes: List['Node']):
        self.nodes = nodes
        self.node_count = len(self.nodes)
        self.dist = [[float('inf')] * self.node_count for _ in range(self.node_count)]
        self.init_dist()
        self.edge_count = 0
        for node in self.nodes:
            self.edge_count += len(node.outputs)

        print("dist:")
        print(self.dist)
        print("edge_count:")
        print(self.edge_count)

    def init_dist(self):
        for i in range(self.node_count):
            self.dist[i][i] = 0
        for node in self.nodes:
            for edge in node.outputs:
                self.add_edge(node, edge.dst_node, edge.w)

    def exist_edge(self, n0: 'Node', n1: 'Node') -> bool:
        assert n0 in self.nodes, "n0 not in graph!"
        assert n1 in self.nodes, "n1 not in graph!"
        for edge in n0.outputs:
            if n1 == edge.dst_node:
                return True
        return False

    def update_dist(self, u, v, w):
        dist = self.dist.copy()
        if dist[u][v] > w:
            dist[u][v] = w
    
        for i in range(self.node_count):
            for j in range(self.node_count):
                if dist[i][v] > dist[i][u] + w:
                    dist[i][v] = dist[i][u] + w
                if dist[u][j] > w + dist[v][j]:
                    dist[u][j] = w + dist[v][j]
                if dist[i][j] > dist[i][u] + w + dist[v][j]:
                    dist[i][j] = dist[i][u] + w + dist[v][j]

        # Detect negetive ring
        for i in range(self.node_count):
            for j in range(self.node_count):
                if dist[i][v] > dist[i][u] + w:
                    return None
                if dist[u][j] > w + dist[v][j]:
                    return None
                if dist[i][j] > dist[i][u] + w + dist[v][j]:
                    return None
        return dist


    def add_edge(self, n0, n1, w) -> bool:
        self.dist[n0.id][n1.id] = w
        dist = self.update_dist(n0.id, n1.id, w)
        if dist is None:
            print("Invalid, negetive ring detected.")
            return False # Invalid   
        self.dist = dist
        print("updated dist:")
        print(self.dist)
        return True

    def del_edge(self, n0, n1):
        pass

    def check_legality(self) -> bool:
        return True

def print_graph(nodes: List['Node']):
    for node in nodes:
        print(node)
        for edge in node.inputs:
            print(edge)
        print('-'*100)

def get_path_value(n0: 'Node', n1: 'Node') -> int:

    return 


import networkx as nx
from graph import *
import json

def load_model(fname: str) -> List[Node]:
    graph = nx.DiGraph()
    with open(fname) as f:
        a = json.load(f)
    node_map = {item[0] : None for item in a}
    ordered_nodes = []
    for node_id, name, sync_type, hw_usage, buffer_layer, is_output, inputs in a:
        input_list = []
        for src_node, src_id in inputs:
            if src_node not in node_map:
                input_list.append(None)
            else:
                assert node_map[src_node] is not None, "Detected ring in topo order {}->{} !".format(src_node, node_id)
                input_list.append([node_map[src_node], src_id])
        node = Node(node_id, input_list, name, sync_type, hw_usage, buffer_layer)
        for src_node, _ in inputs:
            assert node_map[src_node] is not None
            graph.add_edge(node_map[src_node], node, weight=0, type="data")
        node_map[node_id] = node
        ordered_nodes.append(node)
    return ordered_nodes, graph