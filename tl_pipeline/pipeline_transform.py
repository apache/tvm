import json
from graph import *

import networkx as nx
from graph import *
import json
# from pipeline_transform import load_model



def is_edge_valid(graph: 'nx.DiGraph', u: 'Node', v: 'Node', weight: int) -> bool:
    if graph.has_edge(u, v):
        return False
    g = graph.copy()
    g.add_edge(u, v, weight=weight)
    if nx.negative_edge_cycle(g):
        return False
    shortest_paths = dict(nx.all_pairs_bellman_ford_path_length(g, weight='weight'))

    # print("-"*100)
    # print("Edges:", g.edges(data=True))
    for s, d, data in g.edges(data=True):
        edge_weight = data['weight']
        s_to_d_dist = shortest_paths[s][d]

        # print("edge_weight:", edge_weight, "s:", s, "d:", d)
        if edge_weight > s_to_d_dist:
            # print("is_edge_valid failed 0")
            return False
        
        if edge_weight == s_to_d_dist:
            _g = g.copy()
            _g.remove_edge(s, d)
            if nx.has_path(_g, s, d):
                new_s_to_d_dist = nx.bellman_ford_path_length(_g, s, d, weight='weight')
                if new_s_to_d_dist == edge_weight:
                    return False
    return True
    
def is_graph_valid(graph: 'nx.DiGraph', original_graph: 'nx.DiGraph'):
    output_nodes = [node for node in original_graph.nodes() if original_graph.out_degree(node) == 0]
    input_nodes = [node for node in original_graph.nodes() if original_graph.in_degree(node) == 0]
    for output_node in output_nodes:
        for input_node in input_nodes:
            if not nx.has_path(graph, output_node, input_node):
                return False
    return True

def add_backward_edges(graph: 'nx.DiGraph', original_graph: 'nx.DiGraph', ordered_nodes: List['Node'], stream: int, cur_id: int, dst_id: int, result: List['nx.DiGraph']):
    if cur_id == len(ordered_nodes):
        if is_graph_valid(graph, original_graph):
            result.append(graph.copy())
        return
    
    if ordered_nodes[cur_id]._sync_type == "sync":
        add_backward_edges(graph, original_graph, ordered_nodes, stream, cur_id + 1, 0, result)
        return

    if dst_id == len(ordered_nodes):
        add_backward_edges(graph, original_graph, ordered_nodes, stream, cur_id + 1, 0, result)
        return
    
    u = ordered_nodes[cur_id]
    v = ordered_nodes[dst_id]
    if u in list(nx.ancestors(original_graph, v)) or u == v:
        add_backward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
        return
    
    common_hw = [h for h in u.hw_usage if h in v.hw_usage]
    if len(common_hw) == 0 and u.buffer_layer[0] != v.buffer_layer[1]:
        add_backward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
        return
    
    add_backward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
    for w in range(1, stream + 1):
        if is_edge_valid(graph, u, v, w):
            graph.add_edge(u, v, weight=w, type="control")
            add_backward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
            graph.remove_edge(u, v)

def add_forward_edges(graph: 'nx.DiGraph', original_graph: 'nx.DiGraph', ordered_nodes: List['Node'], stream: int, cur_id: int, dst_id: int, result: List['nx.DiGraph']):
    if cur_id == len(ordered_nodes):
        if is_graph_valid(graph, original_graph):
            result.append(graph.copy())
        return
    
    if ordered_nodes[cur_id]._sync_type == "sync":
        add_forward_edges(graph, original_graph, ordered_nodes, stream, cur_id + 1, 0, result)
        return

    if dst_id == len(ordered_nodes):
        add_forward_edges(graph, original_graph, ordered_nodes, stream, cur_id + 1, 0, result)
        return
    
    u = ordered_nodes[cur_id]
    v = ordered_nodes[dst_id]
    if u in list(nx.descendants(original_graph, v)) or u == v:
        add_forward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
        return
    
    common_hw = [h for h in u.hw_usage if h in v.hw_usage]
    if len(common_hw) == 0 and u.buffer_layer[0] != v.buffer_layer[1]:
        add_forward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
        return
    
    add_forward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
    for w in range(-stream, 0):
        if is_edge_valid(graph, u, v, w):
            graph.add_edge(u, v, weight=w, type="control")
            add_forward_edges(graph, original_graph, ordered_nodes, stream, cur_id, dst_id + 1, result)
            graph.remove_edge(u, v)

def transform(graph: 'nx.DiGraph', original_graph: 'nx.DiGraph', topo_ordered_nodes: List['Node'], stream: int):
    results = []
    interm_results = []
    add_backward_edges(graph, original_graph, list(reversed(topo_ordered_nodes)), stream=stream, cur_id=0, dst_id=0, result=interm_results)
    for modified_graph in interm_results:
        add_forward_edges(modified_graph, original_graph, list(topo_ordered_nodes), stream=stream, cur_id=0, dst_id=0, result=results)
    return results

if __name__ == "__main__":
    _, graph = load_model("inputs.json")
    print(graph)
    print("Nodes:", graph.nodes(data=True))
    print("Edges:", graph.edges(data=True))

    ordered_nodes = list(nx.topological_sort(graph))
    print("ordered_nodes:", ordered_nodes)

    results = []
    add_backward_edges(graph, graph.copy(), list(reversed(ordered_nodes)), stream=2, cur_id=0, dst_id=0, result=results)
    for i, g in enumerate(results):
        print(f"Graph {i+1}:")
        print(g.edges(data=True))