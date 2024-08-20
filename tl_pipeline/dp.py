import networkx as nx
from graph import *

class Schedule:
    def __init__(self, order, hw_time):
        self.order = order
        self.hw_time = {}
        self.hw_time["tensor_core"] = hw_time["tensor_core"]
        self.hw_time["cuda_core"] = hw_time["cuda_core"]
        self.hw_time["tma"] = hw_time["tma"]

    @property
    def time(self) -> float:
        return max(self.hw_time["tensor_core"], self.hw_time["cuda_core"], self.hw_time["tma"])

    def __repr__(self) -> str:
        nodes_str = [repr(node) for node in self.order]
        s = ", ".join(nodes_str)
        s += "\n"
        s += f"Time:{self.time}"
        return s
    
dp_dict = {}    

def get_schedule(nodes: List[Node], in_node: Node, in_node_dsts: List[Node]) -> Schedule:
    schedule = dp_dict[frozenset(nodes)]
    new_order = [in_node] + schedule.order
    end_time = schedule.hw_time[in_node.hw_usage]
    for dst_node in in_node_dsts:
        if schedule.hw_time[dst_node.hw_usage] > end_time:
            end_time = schedule.hw_time[dst_node.hw_usage]
    if in_node._sync_type == "sync":
        end_time = schedule.time
    start_time = end_time + in_node.time
    # start time can not be later than any node in nodes
    if start_time < schedule.time:
        start_time = schedule.time
    
    new_hw_time = (schedule.hw_time).copy()
    new_hw_time[in_node.hw_usage] = start_time
    return Schedule(new_order, new_hw_time)

def dp(graph: Graph, nodes: List[Node]):
    global dp_dict
    pred_nodes = {}
    
    subgraph = graph.subgraph(nodes)
    for node in subgraph.nodes:
        for pred in graph.predecessors(node):
            valid = True
            if pred not in subgraph:
                for pred_succ in graph.successors(pred):
                    if pred_succ not in subgraph:
                        valid = False
                        break
                if valid:
                    pred_nodes[pred] = [succ for succ in graph.successors(pred) if succ in subgraph]
    # for node in pred_nodes:
    #     print(node)
    for in_node, in_node_dsts in pred_nodes.items():
        schedule = get_schedule(nodes, in_node, in_node_dsts)
        new_nodes = nodes + [in_node]
        new_nodes.append(in_node)
        if frozenset(new_nodes) in dp_dict and dp_dict[frozenset(new_nodes)].time < schedule.time:
            continue
        dp_dict[frozenset(new_nodes)] = schedule
        dp(graph, nodes + [in_node])

def duplicate_with_stream(graph: Graph, stream: int) -> Tuple[Graph, Node]:
    node_num = len(graph.nodes)
    node_list = []
    duplicate_graph = nx.DiGraph()
    for i in range(stream):
        mapping = {node: Node(node.id, [], node.name+f'_{i}', node._sync_type, node.hw_usage, node.time, node.buffer_layer) for node in graph.nodes}
        g = nx.relabel_nodes(graph, mapping)
        node_list.extend(g.nodes)
        duplicate_graph = nx.compose(duplicate_graph, g)
        
        if i > 0:
            for nid in range(node_num):
                duplicate_graph.add_edge(node_list[(i - 1) * node_num + nid], node_list[i * node_num + nid], type="control.issue")
    
    out_nodes = [node for node in duplicate_graph.nodes if duplicate_graph.out_degree(node) == 0]
    output_node = Node(node_num * stream, [], "Output", "sync", "cuda_core", 0, [])
    
    for out_node in out_nodes:
        duplicate_graph.add_edge(out_node, output_node, type="data")
    return duplicate_graph, output_node

if __name__ == "__main__":
    stream = 3
    _, graph = load_model("inputs.json")
    duplicate_graph, output_node = duplicate_with_stream(graph, stream)

    print(duplicate_graph)
    print(duplicate_graph.edges(data=True))
    # for node in graph.nodes:
    #     print(node)
    dp_dict[frozenset([output_node])] = Schedule([output_node], {"tensor_core":0, "cuda_core":0, "tma":0})
    dp(duplicate_graph, [output_node])

    print("dp_dict", dp_dict)
    print("results", dp_dict[frozenset(list(duplicate_graph.nodes))])