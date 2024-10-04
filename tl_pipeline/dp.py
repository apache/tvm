import networkx as nx
from graph import *
from functools import lru_cache

class Schedule:
    def __init__(self, order: List[Node], sync: List[Tuple[Node, Node]], hw_time: Dict):
        self.order = order
        self.sync = sync
        self.hw_time = {}
        self.hw_time["tensor_core"] = hw_time["tensor_core"]
        self.hw_time["cuda_core"] = hw_time["cuda_core"]
        self.hw_time["tma"] = hw_time["tma"]

    @property
    def time(self) -> float:
        return max(self.hw_time["tensor_core"], self.hw_time["cuda_core"], self.hw_time["tma"])

    def __lt__(self, other):
        return isinstance(other, Schedule) and (self.order != other.order or self.sync != other.sync) and self.time < other.time
    
    def __eq__(self, other):
        return isinstance(other, Schedule) and self.order == other.order and self.sync == other.sync

    def __hash__(self):
        return hash((tuple(self.order), tuple(tuple(pair) for pair in self.sync)))
    
    def __repr__(self) -> str:
        nodes_str = [repr(node) for node in self.order]
        sync_str = ["(" + repr(n0) + ", " + repr(n1) + ")" for n0, n1 in self.sync]
        # s = f"len={len(nodes_str)}, "
        s = ""
        s += ", ".join(nodes_str)
        s += "\n"
        s += ",".join(sync_str)
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
smem_interval = 1024
reg_interval = 1024
smem_max_cap = 4 * 1024
reg_max_cap = 4 * 1024
issue_latency = 0.001

def get_schedule(nodes: List[Node], in_node: Node, graph: nx.DiGraph, mem_cap: List[float]) -> List[Schedule]:
    schedule_list = []
    smem_cap, reg_cap = mem_cap
    schedules = []
    for s in range(int(smem_cap / smem_interval)):
        for r in range(int(reg_cap / reg_interval)):
            schedules.extend(dp_dict[frozenset(nodes)][s][r])
    for schedule in schedules:
        new_order = [in_node] + schedule.order
        new_hw_time = {"tensor_core":0, "cuda_core":0, "tma":0}
        # node: (start_time, end_time)
        issue_dict = {} 
        last_start_time = 0
        for cur_node in new_order:
            start_time = max(new_hw_time[cur_node.hw_usage], last_start_time + issue_latency)
            for pred_node in new_order:
                # Need to issue after previous sync node done
                if pred_node in issue_dict and pred_node.sync_type == "sync":
                    start_time = max(start_time, issue_dict[pred_node][1] + issue_latency)
                if graph.has_edge(pred_node, cur_node):
                    edge = graph.get_edge_data(pred_node, cur_node)
                    if edge['type'] == "data":
                        assert pred_node in issue_dict, "Error: pred_node not in issue_dict"
                        # Data denpendency
                        start_time = max(start_time, issue_dict[pred_node][1] + issue_latency)
            issue_dict[cur_node] = [start_time, start_time + cur_node.time]
            new_hw_time[cur_node.hw_usage] = start_time + cur_node.time
            last_start_time = start_time
        
        new_schedule = Schedule(new_order, schedule.sync, new_hw_time)
        smem_fp, reg_fp = calculate_memory_footprint(new_schedule, graph)

        if smem_fp <= smem_cap and reg_fp <= reg_cap:
            schedule_list.append(new_schedule)

        # Add syncs conditions for async nodes
        if in_node.sync_type == "async":
            for sync_dst_node in schedule.order:
                if sync_dst_node.name == "Output":
                    continue
                # If no hardware resource conflict, no need to add sync
                if sync_dst_node.hw_usage != in_node.hw_usage and sync_dst_node.output_buffer not in in_node.input_buffer:
                    continue
                new_schedule = Schedule(new_order, schedule.sync + [(in_node, sync_dst_node)], new_hw_time)
                smem_fp, reg_fp = calculate_memory_footprint(new_schedule, graph)
                if smem_fp <= smem_cap and reg_fp <= reg_cap:
                    schedule_list.append(new_schedule)
    return sorted(schedule_list)

@lru_cache(None)
def get_data_dependency_inputs(node: Node, graph: nx.DiGraph) -> List[Node]:
    return [pred for pred in graph.predecessors(node) if graph.get_edge_data(pred, node).get('type') == 'data']

@lru_cache(None)
def get_data_dependency_outputs(node: Node, graph: nx.DiGraph) -> List[Node]:
    return [succ for succ in graph.successors(node) if graph.get_edge_data(node, succ).get('type') == 'data']

def calculate_memory_footprint(schedule: Schedule, graph: nx.DiGraph) -> List[float]:
    smem_fp = 0
    reg_fp = 0
    smem_usage = 0
    reg_usage = 0

    def allocate_outputs(node: Node):
        nonlocal smem_usage, reg_usage
        if node.output_buffer == "smem":
            smem_usage += node.dsize
        elif node.output_buffer == "register":
            reg_usage += node.dsize

    def free_outputs(node: Node):
        nonlocal smem_usage, reg_usage
        if node.output_buffer == "smem":
            smem_usage -= node.dsize
        elif node.output_buffer == "register":
            reg_usage -= node.dsize
    
    ref_dict = {node: [] for node in schedule.order}
    for ordered_node in schedule.order:
        for pred in get_data_dependency_inputs(ordered_node, graph):
            if pred in schedule.order:
                ref_dict[pred].append(ordered_node)
    for ordered_node in schedule.order:
        sync_srcs = [src for src, dst in schedule.sync if dst == ordered_node]
        for pred in list(set(get_data_dependency_inputs(ordered_node, graph) + sync_srcs)):
            if pred in schedule.order:
                for pred_src in get_data_dependency_inputs(pred, graph):
                    if pred_src in schedule.order:
                        if len(ref_dict[pred_src]) == 0:
                            continue
                        ref_dict[pred_src].remove(pred)
                        if len(ref_dict[pred_src]) == 0:
                            free_outputs(pred_src)
        allocate_outputs(ordered_node)
        if smem_usage > smem_fp:
            smem_fp = smem_usage
        if reg_usage > reg_fp:
            reg_fp = reg_usage
    # print(smem_fp, reg_fp)
    return [smem_fp, reg_fp]

def dp(graph: nx.DiGraph, prev_sub_graphs: List[List[Node]]):
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
        dp_dict[new_nodes_set] = []
        for s in range(smem_max_cap // smem_interval):
            smem_cap = smem_interval * (s + 1)
            dp_dict[new_nodes_set].append([])
            for r in range(reg_max_cap // reg_interval):
                reg_cap = reg_interval * (r + 1)
                schedule_list = []
                for in_node, prev_nodes in nodes:
                    schedule_list.extend(get_schedule(prev_nodes, in_node, graph, [smem_cap, reg_cap]))
                dp_dict[new_nodes_set][s].append(sorted(set(schedule_list))[:topk])

    sub_graphs = list(list(n) for n in list(sub_graphs_dict))
    if len(sub_graphs) > 0:
        dp(graph, sub_graphs)


def duplicate_with_stream(graph: nx.DiGraph, stream: int) -> Tuple[nx.DiGraph, Node]:
    node_num = len(graph.nodes)
    node_list = []
    duplicate_graph = nx.DiGraph()
    for i in range(stream):
        mapping = {node: Node(node.id, node.name+f'_{i}', node.sync_type, node.hw_usage, node.time, node.input_buffer, node.output_buffer) for node in graph.nodes}
        g = nx.relabel_nodes(graph, mapping)
        ordered_ndoes = sorted(g.nodes)
        node_list.extend(ordered_ndoes)
        duplicate_graph = nx.compose(duplicate_graph, g)
        
        if i > 0:
            for nid in range(node_num):
                duplicate_graph.add_edge(node_list[(i - 1) * node_num + nid], node_list[i * node_num + nid], type="control.issue")
    
    out_nodes = [node for node in duplicate_graph.nodes if duplicate_graph.out_degree(node) == 0]
    output_node = Node(node_num * stream, "Output", "sync", "cuda_core", 0, ["register"], "register")
    
    for out_node in out_nodes:
        duplicate_graph.add_edge(out_node, output_node, type="data")
    return duplicate_graph, output_node

if __name__ == "__main__":
    stream = 3
    graph = load_model("inputs.json")
    duplicate_graph, output_node = duplicate_with_stream(graph, stream)

    print(duplicate_graph)
    print(duplicate_graph.edges(data=True))
    # for node in graph.nodes:
    #     print(node)
    dp_dict[frozenset([output_node])] = []
    for s in range(smem_max_cap // smem_interval):
        dp_dict[frozenset([output_node])].append([])
        for r in range(reg_max_cap // reg_interval):
            dp_dict[frozenset([output_node])][s].append([Schedule([output_node], [], {"tensor_core":0, "cuda_core":0, "tma":0})])
    dp(duplicate_graph, [[output_node]])

    # for k, v in dp_dict.items():
    #     print(k)
    #     print(v)
    print("results:")
    for schedule in dp_dict[frozenset(list(duplicate_graph.nodes))]:
        print(schedule)