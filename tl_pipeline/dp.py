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

    def __lt__(self, other):
        return isinstance(other, Schedule) and self.order != other.order and self.time < other.time
    
    def __eq__(self, other):
        return isinstance(other, Schedule) and self.order == other.order

    def __hash__(self):
        return hash(tuple(self.order))
    
    def __repr__(self) -> str:
        nodes_str = [repr(node) for node in self.order]
        # s = f"len={len(nodes_str)}, "
        s = ""
        s += ", ".join(nodes_str)
        s += "\n"
        s += f"Time:{self.time}, "
        s += f"tensor_core:{self.hw_time['tensor_core']}, "
        s += f"cuda_core:{self.hw_time['cuda_core']}, "
        s += f"tma:{self.hw_time['tma']}, "
        s += "\n"
        return s
    
dp_dict = {}    
topk = 10
i = 0
issue_latency = 0.001

def get_schedule(nodes: List[Node], in_node: Node, in_node_dsts: List[Node], graph: Graph) -> List[Schedule]:
    schedule_list = []
    for schedule in dp_dict[frozenset(nodes)]:
        new_order = [in_node] + schedule.order
        new_hw_time = {"tensor_core":0, "cuda_core":0, "tma":0}
        # node: (start_time, end_time)
        issue_dict = {} 
        last_start_time = 0
        for cur_node in new_order:
            start_time = max(new_hw_time[cur_node.hw_usage], last_start_time + issue_latency)
            for pred_node in new_order:
                # Need to issue after previous sync node done
                if pred_node in issue_dict and pred_node._sync_type == "sync":
                    start_time = max(start_time, issue_dict[pred_node][1] + issue_latency)
                if graph.has_edge(pred_node, cur_node):
                    edge = graph.get_edge_data(pred_node, cur_node)
                    if edge['type'] == "data":
                        assert pred_node in issue_dict, "error: pred_node not in issue_dict"
                        # Data denpendency
                        start_time = max(start_time, issue_dict[pred_node][1] + issue_latency)
            issue_dict[cur_node] = [start_time, start_time + cur_node.time]
            new_hw_time[cur_node.hw_usage] = start_time + cur_node.time
            last_start_time = start_time
        schedule_list.append(Schedule(new_order, new_hw_time))
    return sorted(schedule_list)


def dp(graph: Graph, prev_sub_graphs: List[List[Node]]):
    global dp_dict
    global i
    # set(sub_graph_nodes): [in_node_0, in_node_1, ...]
    sub_graphs_dict = {}
    for prev_nodes in prev_sub_graphs:
        subgraph = graph.subgraph(prev_nodes.copy())
        for node in subgraph.nodes:
            for pred in graph.predecessors(node):
                valid = True
                if pred not in subgraph:
                    for pred_succ in graph.successors(pred):
                        if pred_succ not in subgraph:
                            valid = False
                            break
                    if valid:
                        key = frozenset(prev_nodes + [pred])
                        if key not in sub_graphs_dict:
                            sub_graphs_dict[key] = []
                        sub_graphs_dict[key].append((pred, prev_nodes))
    for new_nodes_set, nodes in sub_graphs_dict.items():
        schedule_list = []
        for in_node, prev_nodes in nodes:
            schedule_list.extend(get_schedule(prev_nodes, in_node, [], graph))
        dp_dict[new_nodes_set] = sorted(set(schedule_list))[:topk]
    sub_graphs = list(list(n) for n in list(sub_graphs_dict))
    if len(sub_graphs) > 0:
        dp(graph, sub_graphs)


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
    dp_dict[frozenset([output_node])] = [Schedule([output_node], {"tensor_core":0, "cuda_core":0, "tma":0})]
    dp(duplicate_graph, [[output_node]])

    # for k, v in dp_dict.items():
    #     print(k)
    #     print(v)
    print("results:")
    for schedule in dp_dict[frozenset(list(duplicate_graph.nodes))]:
        print(schedule)