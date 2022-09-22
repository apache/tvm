from asyncore import read
from enum import Enum
import networkx as nx

# TODO: save at another file
X220 = {
    # Scalar
    'Immediate': 1,
    'Variable': 1,
    'Add': 1,

    # Memory
    'Store': 6,
    'Load': 6,

    # Vector
    'LT': 3,
    'Mul': 3,
    'Max': 3,
}

## Slot
# {'Scalar': None, ...} -> Empty
# {'Scalar': {'name': 'XXX', 'start': 0, 'end': 2}, ...} -> Occupied @0~1, Freed at @2

## Graph Node
# {node_num: {'name': '~~~', slot': 'Memory', 'type': 'Load'}, ...}

class VectorProcessor():
    def __init__(self, model, graph):
        self.model = model
        self.slot  = Slot(model)
        self.instr = Instruction(model)
        
        if not isinstance(graph, nx.classes.digraph.DiGraph):
            raise TypeError("%s NOT supported as input"%(type(graph)))
        self.graph = graph
        self.source = self.get_source(self.graph)
        self.sink = self.get_sink(self.graph)

        self.node = Nodes(graph)
        
        # self.in_nodes = self.graph.nodes
        # self.candid_nodes = []
        # self.out_nodes = []

    ## FIXME: split graph to another class
    def get_source(self, G):
        return [node for node in G.nodes if G.in_degree(node)==0]
    def get_sink(self, G):
        return [node for node in G.nodes if G.out_degree(node)==0]
    ######################################

    ##
    def slot_free(self, clk):
        free_nodes = self.slot.free(clk)
        free_nodes = list(dict.fromkeys(free_nodes)) # Remove duplicates in list ('Control')
        return free_nodes

    def slot_occupy(self, ready_nodes, clk):
        if len(ready_nodes)==1 and self.graph.nodes[ready_nodes[0]]['type']=='KERNEL END':
            return True
        else:
            for rn in ready_nodes:
                if self.graph.nodes[rn]['slot'] == 'Control':
                    if all([self.slot.is_idle(slot) for slot in self.slot.get_keys()]):
                        [self.slot.occupy(self.graph, rn, slot, clk) for slot in self.slot.get_keys()]
                    else:
                        continue
                elif self.slot.is_idle(self.graph.nodes[rn]['slot']):
                    self.slot.occupy(self.graph, rn, self.graph.nodes[rn]['slot'], clk)
            return False

    def node_update(self, free_nodes):
        # free_nodes: Remove from rdy_nodes & Add to out_node
        for fn in free_nodes:
            self.node.free(fn)
        # Newly ready nodes: Remove from in_nodes & Add to rdy_nodes 
        for fn in free_nodes:
            for s in self.graph.successors(fn):
                if set(self.graph.predecessors(s)).issubset(set(self.node.get_out_nodes())):
                    self.node.ready(s)
    ##

    def add_END_node_to_graph(self):
        sinks = [node for node in self.graph.nodes if self.graph.out_degree(node)==0]
        if len(sinks) > 1: raise NotImplementedError("Kernel Graph must end with a single sink node")
        node_num = max(list(self.graph.nodes))+1
        self.graph.add_node(node_num, name='KERNEL END', slot='Control', type='KERNEL END')
        self.graph.add_edge(sinks[0], node_num)

    def add_weight_to_graph(self):
        for node in self.graph.nodes:
            childs = self.graph.successors(node)
            for child in childs:
                if self.graph.nodes[node]['slot'] == 'Control':
                    self.graph[node][child]['weight'] = 1 # latency=1 for control nodes
                else:
                    self.graph[node][child]['weight'] = self.instr.get_weight(self.graph.nodes[node]['type'])
        return self.graph

    '''
    [Rules for candidate nodes]
     1. A node can be executed only after ALL parent nodes are executed.
     2. A node can be executed when slot is available
    '''
    def run_model(self):
        '''    
        [Steps to Run Model]
        1. Update each slot's state for current CLK
        2. For available slot, occupy it with node in candid_nodes with appropriate type
        3. Update node states (in_nodes, candid_nodes, out_nodes)
        4. Update CLK
        '''

        # Initialize when clk=0
        clk = 0
        self.node.update_rdy_nodes(self.source)
        self.node.update_in_nodes(list(set(self.node.get_in_nodes())-set(self.node.get_rdy_nodes()))) # Update in_nodes

        while True:
            free_nodes = self.slot_free(clk)
            self.node_update(free_nodes)
            ready_nodes = self.node.get_rdy_nodes()
            END = self.slot_occupy(ready_nodes, clk)
            print(clk)
            print(">> in_nodes: %s\n   rdy_nodes: %s\n   out_nodes: %s"
                    %(self.node.get_in_nodes(), self.node.get_rdy_nodes(), self.node.get_out_nodes()))
            self.slot.print_status()
            print("#"*20)
            if END: break

            clk = clk+1

        print("Total Run Time: %d"%(clk))


class Slot():
    def __init__(self, model):
        if model == "X220":
            self.slots = dict.fromkeys(['Scalar', 'Memory', 'Vector'])
        else:
            raise NotImplementedError("Currently NOT supported model: %s" %(model))

    def free(self, clk):
        free_nodes = []
        for type in self.slots.keys():
            if self.is_empty(type):
                continue
            if self.slots[type]['end'] == clk:
                free_nodes.append(self.slots[type]['num'])
                self.slots[type] = None
        return free_nodes

    def is_empty(self, type):
        return self.slots[type] == None

    def occupy(self, graph, node_num, slot, clk):
        dur = max([graph[node_num][child]['weight'] for child in graph.successors(node_num)])
        self.slots[slot] = {
            # 'node': node,
            'num': node_num,
            'start': clk, 
            'end': clk+dur
        }

    def available(self, graph, node_num, out_nodes):
        # IDLE: when slot is idle
        # DEPN: when all parent nodes are exectued
        node = graph.nodes[node_num]
        IDLE = node['slot'] in self.get_idle()
        DEPN = set(graph.predecessors(node_num)).issubset(set(out_nodes))
        return (IDLE and DEPN)

    def get_idle(self):
        return [type for type in self.slots.keys() if self.is_idle(type)]
        
    def is_idle(self, type):
        if type in ['Seq', 'For Start', 'For End', 'END']:
            return all([self.slots[key]==None for key in self.slots.keys()])
        else:
            return self.slots[type] == None

    def get_keys(self):
        return self.slots.keys()

    def print_status(self):
        for key in self.get_keys():
            print("%s: %s"%(key, self.slots[key]))

class Nodes():
    '''
    in_nodes:   Ready: X, Processed: X
    rdy_nodes:  Ready: O, Processed: X
    out_nodes:  Ready: O, Processed: O
    '''
    def __init__(self, graph):
        self.in_nodes = graph.nodes
        self.rdy_nodes = []
        self.out_nodes = []
    
    def get_nodes(self):
        return self.in_nodes, self.rdy_nodes, self.out_nodes
    def get_in_nodes(self):
        return self.in_nodes
    def get_rdy_nodes(self):
        return self.rdy_nodes
    def get_out_nodes(self):
        return self.out_nodes

    def update_in_nodes(self, nodes):
        self.in_nodes = nodes
    def update_rdy_nodes(self, nodes):
        self.rdy_nodes = nodes
    def update_out_nodes(self, nodes):
        self.out_nodes = nodes

    def free(self, free_node):
        self.rdy_nodes.remove(free_node)
        self.out_nodes.append(free_node)

    def ready(self, rdy_node):
        self.in_nodes.remove(rdy_node)
        self.rdy_nodes.append(rdy_node)

class Instruction():
    def __init__(self, model):
        if model == "X220": 
            self.instr = X220
        else: 
            raise NotImplementedError("Cuurently NOT supported model: %s" %(model))

    def get_weight(self, type):
        return self.instr[type]