from typing import Any, List, Dict, Optional, Tuple, Union

class Edge:
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int, diff: int):
        self.diff = diff
        self.src_node = src_node
        self.dst_node = dst_node
        self.src_id = src_id
        self.dst_id = dst_id
    
    def __repr__(self) -> str:
        return "<Edge, " + self.src_node.name + "--(" + str(self.diff) + ")-->" + self.dst_node.name + ">"

class DataDependencyEdge(Edge):
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int, diff: int):
        super().__init__(src_node, dst_node, src_id, dst_id, diff)

    def __repr__(self) -> str:
        return "<DataDependencyEdge, " + self.src_node.name + "--(" + str(self.diff) + ")-->" + self.dst_node.name + ">"

class AntiDependencyEdge(Edge):
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int, diff: int):
        super().__init__(src_node, dst_node, src_id, dst_id, diff)

    def __repr__(self) -> str:
        return "<AntiDependencyEdge, " + self.src_node.name + "--(" + str(self.diff) + ")-->" + self.dst_node.name + ">"

class CondDependencyEdge(Edge):
    def __init__(self, src_node: 'Node', dst_node: 'Node', src_id: int, dst_id: int, diff: int):
        super().__init__(src_node, dst_node, src_id, dst_id, diff)

    def __repr__(self) -> str:
        return "<CondDependencyEdge, " + self.src_node.name + "--(" + str(self.diff) + ")-->" + self.dst_node.name + ">"

class Node:
    def __init__(self, inputs: List[Union[Tuple['Node', int], 'Node', None]], name: str, sync_type: str):
        self.name = name
        self._out_edges = []
        self._in_edges = []
        self._shapes = []
        self._dtypes = []
        self._tag = {}
        self._sync_type = sync_type
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
            edge = AntiDependencyEdge(self, src_node, 0, 0, -1)
            self._out_edges.append(edge)
            src_node._in_edges.append(edge)

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
        return "<Node, " + self.name + ">"

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


def transform(nodes: List['Node'], stream_num: int, cond_deps) -> List['Node']:
    return nodes

# mha Example
if __name__ == "__main__":
    loadk = Node(inputs=[None], name="loadk", sync_type='async')
    mma0 = Node(inputs=[loadk], name="mma0", sync_type='async')
    loadv = Node(inputs=[None], name="loadv", sync_type='async')
    softmax = Node(inputs=[mma0], name="softmax", sync_type='sync')
    mma1 = Node(inputs=[loadv, softmax], name="mma1", sync_type='async')
    out = OutputNode(mma1)
    ordered_nodes = [loadk, mma0, loadv, softmax, mma1, out]

    for edge in softmax.inputs:
        print(edge)
