from graph import *
from itertools import permutations, product

class Instruction:
    def __init__(self, node: Node, iter: int = -1):
        self.instr = node
        self.iter = iter
      
    def __repr__(self) -> str:
        return repr(self.instr)
    
    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.instr == other.instr and self.iter == other.iter
    
    def __gt__(self, other):
        if type(self) is not type(other):
            return False
        return self.instr == other.instr and self.iter > other.iter
    
    def __lt__(self, other):
        if type(self) is not type(other):
            return False
        return self.instr == other.instr and self.iter < other.iter
    
    def __ge__(self, other):
        if type(self) is not type(other):
            return False
        return self.instr == other.instr and self.iter >= other.iter
    
    def __le__(self, other):
        if type(self) is not type(other):
            return False
        return self.instr == other.instr and self.iter <= other.iter
    
    def __hash__(self):
        return hash((self.instr, self.iter))
    
class Issue(Instruction):
    def __init__(self, node: Node, iter: int = -1):
        super().__init__(node, iter)
  
    def __repr__(self) -> str:
        return super().__repr__() + f".issue({self.iter})"
    
    def __hash__(self):
        return hash((super().__hash__(), 'Issue'))

class Wait(Instruction):
    def __init__(self, node: Node, iter: int = -1):
        super().__init__(node, iter)
   
    def __repr__(self) -> str:
        return super().__repr__() + f".wait({self.iter})"
    
    def __hash__(self):
        return hash((super().__hash__(), 'Wait'))

class Plan:
    def __init__(self, instrs: List[Instruction], graph_id: int = -1) -> None:
        self.instrs = instrs
        self.graph_id = graph_id
    
    def set_graph_id(self, graph_id: int):
        self.graph_id = graph_id

    def __repr__(self) -> str:
        s = ""
        for instr in self.instrs:
            s += repr(instr)
            s += "\n"
        return s
    
    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        if len(self.instrs) != len(other.instrs):
            return False
        for i in range(len(self.instrs)):
            if self.instrs[i] != other.instrs[i]:
                return False
        return True
    
    def __hash__(self):
        return hash(tuple(self.instrs))

def is_valid_graph(graph: 'nx.DiGraph') -> bool:
    if nx.negative_edge_cycle(graph):
        return False
    return True

def get_plan(graph: 'nx.DiGraph', updated_graph: 'nx.DiGraph', ordered_nodes: List['Node']) -> Union[List['Instruction'], None]:
    if not is_valid_graph(updated_graph):
        return None
    instrs = []
    
    for node in ordered_nodes:
        iteration = nx.bellman_ford_path_length(updated_graph, ordered_nodes[0], node)
        for pre_node in list(graph.predecessors(node)):
            if pre_node._sync_type == "sync":
                continue
            wait_instr = Wait(pre_node, iteration - graph[pre_node][node]['weight'])
            flag = True
            for instr in instrs:
                # Unnecessary redundant waiting
                if instr >= wait_instr:
                    flag = False
                    break
                # Example: 
                # mma.wait(0)
                # ...
                # mma.wait(1)
                # Then mma.wait(0) can be removed because this wait is done in the previous iteration
                if instr < wait_instr:
                    instrs.remove(instr)
            if flag:
                instrs.append(wait_instr)
        instrs.append(Issue(node, iteration))
    # for r in instrs:
    #     print(r)
    # print("="*100)
    return Plan(instrs)

def update_base_on_order(graph: 'nx.DiGraph', ordered_nodes: List['Node']) -> 'nx.DiGraph':
    g = graph.copy()
    for s, d, data in g.edges(data=True):
        if ordered_nodes.index(s) > ordered_nodes.index(d):
            g[s][d]['weight'] = data['weight'] - 1
    return g

def generate(graph: 'nx.DiGraph', topo_ordered_nodes: List['Node']) -> Union[List[List['Instruction']], None]:
    # We only support tile-graph with 1 output node
    # If there are 2 output nodes in tile-graph, you can add a new output node and add edge to this new node 
    # topo_ordered_nodes = list(nx.topological_sort(graph))
    output_nodes = []
    for node in graph.nodes():
        is_output = True
        for _, _, data in graph.out_edges(node, data=True):
            if data.get('type') == 'data':
                is_output = False
                break
        if is_output:
            output_nodes.append(node)
    assert len(output_nodes) == 1, "Error: number of output_node is not 1."
    results = []
    all_orders = list(permutations(range(len(graph.nodes) - 1)))
    # print(all_orders)
    
    for order in all_orders:
        ordered_nodes = [topo_ordered_nodes[-1]]
        for i in range(len(graph.nodes) - 1):
            ordered_nodes.append(topo_ordered_nodes[order[i]])
        updated_graph = update_base_on_order(graph, ordered_nodes)
        plan = get_plan(graph, updated_graph, ordered_nodes)
        if plan is not None:
            results.append(plan)
    return results

if __name__ == "__main__":
    _, graph = load_model("inputs.json")
    topo_ordered_nodes = list(nx.topological_sort(graph))
    print("topo_ordered_nodes:", topo_ordered_nodes)

    v0, v3, v1, v2, v4 = topo_ordered_nodes
    graph.add_edge(v4, v1, weight=2, type="control")
    graph.add_edge(v4, v3, weight=2, type="control")
    graph.add_edge(v1, v0, weight=1, type="control")

    print(graph)
    print("Nodes:", graph.nodes(data=True))
    print("Edges:", graph.edges(data=True))

    plans = generate(graph, topo_ordered_nodes)

    for i, plan in enumerate(plans):
        print("-" * 100)
        print(f"Plan {i}:")
        for instr in plan:
            print(instr)
