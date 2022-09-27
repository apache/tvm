import networkx as nx
import tvm_config

class VectorProcessor():
    def __init__(self, model, graph, debug):
        if not model in ['x220', 'x330']:
            raise NotImplementedError("Currently Not Supported Model: %s"%(model))
        if not isinstance(graph, nx.digraph.DiGraph):
            raise TypeError("Currently Not Supported Graph: %s"%(type(graph)))

        self.model = model
        self.graph = graph
        self.slot = dict.fromkeys(self.import_slot(model))
        self.latency = self.import_latency(model)
        self.type = self.import_type(model)
        self.debug = debug

    def import_slot(self, model):
        return tvm_config.Config[model]['Slot']

    def import_latency(self, model):
        return tvm_config.Config[model]['Latency']

    def import_type(self, model):
        return tvm_config.Config[model]['Type']

    def color_graph(self):
        misc_nodes = list(self.graph.nodes)

        #1. Color distinct nodes
        for node in self.graph.nodes:
            node_type = self.graph.nodes[node]['type']
            for slot in self.type.keys():
                if node_type in self.type[slot]: 
                    self.graph.nodes[node]['slot'] = slot
                    misc_nodes.remove(node)

        #2. Color Miscellaneous nodes
        while len(misc_nodes): # iterate for remaining nodes
            for misc in misc_nodes:
                parents = list(self.graph.predecessors(misc))
                # Color node if all parent's slot is colored appropriately
                if all(['slot' in self.graph.nodes[parent] for parent in parents]) and \
                  all([self.graph.nodes[parent]['slot'] in self.slot.keys() for parent in parents]):
                    parent_slots = []
                    for parent in parents:
                        parent_slots.append(self.graph.nodes[parent]['slot'])
                    if all(s == 'Scalar' for s in parent_slots): # (S, S) --> (S)
                        self.graph.nodes[misc]['slot'] = 'Scalar' 
                    elif all(s in ['Memory', 'Vector'] for s in parent_slots): # (M, M), (M, V), (V, V) --> (V)
                        self.graph.nodes[misc]['slot'] = 'Vector'
                    else:
                        raise RuntimeError("Currently Not Supported MISC node's parent type: %s"%(parent_slots))
                    misc_nodes.remove(misc)
                    break
        return

    def run_model(self):
        '''    
        [Steps in 'run_model()']
        1. Update each slot's state for current CLK
        2. For available slot, occupy it with node in candid_nodes with appropriate type
        3. Update node states (in_nodes, candid_nodes, out_nodes)
        4. Update CLK
        '''

        '''
        [Nodes Status]
        in_nodes:   Ready: X, Processed: X
        rdy_nodes:  Ready: O, Processed: X
        out_nodes:  Ready: O, Processed: O
        '''
        in_nodes, rdy_nodes, out_nodes = [], [], []

        # Initialize: clk=0
        clk = 0
        rdy_nodes = [node for node in self.graph.nodes if self.graph.in_degree(node)==0]
        in_nodes = list(set(self.graph.nodes)-set(rdy_nodes)) # Update in_nodes

        while True:
            free_nodes = self.slot_update(clk)
            in_nodes, rdy_nodes, out_nodes = self.node_update(free_nodes, in_nodes, rdy_nodes, out_nodes)
            if len(in_nodes)==0 and len(rdy_nodes)==1:
                break
            self.slot_occupy(in_nodes, rdy_nodes, clk)
            self.print_status(clk, in_nodes, rdy_nodes, out_nodes)
            clk = clk+1

        self.print_status(clk, in_nodes, rdy_nodes, out_nodes)
        print("Total Run Time: %d"%(clk))

    def slot_update(self, clk):
        free_nodes = []
        #1. Free slot w/ finished op & Add finished op to 'free_nodes'
        for s in self.slot.keys():
            if (not self.slot[s]==None) and (self.slot[s]['end']==clk):
                free_nodes.append(self.slot[s]['node'])
                self.slot[s] = None

        #2. Remove duplicates in list (; All three slots are occupied by 'Control')
        free_nodes = list(dict.fromkeys(free_nodes))

        return free_nodes

    def slot_occupy(self, in_nodes, rdy_nodes, clk):
        for rn in rdy_nodes:
            slot = self.graph.nodes[rn]['slot']
            type = self.graph.nodes[rn]['type']
            try:
                dur = self.latency[slot][type]
            except:
                raise TypeError("Currently NOt Supported latency config; Slot: %s, Type: %s"%(slot, type))
            if slot == 'Control':
                if all([self.slot[s]==None for s in self.slot.keys()]): # If all slots are IDLE, 'Occupy'!!
                    for s in self.slot.keys():
                        self.slot[s] = {'node': rn, 'start': clk, 'end': clk+dur} # Occupy
                else: # Else, move to next clk
                    continue
            else:
                if self.slot[slot] == None:
                    self.slot[slot] = {'node': rn, 'start': clk, 'end': clk+dur} # Occupy
                else:
                    continue


    def node_update(self, free_nodes, in_nodes, rdy_nodes, out_nodes):
        #1. free_nodes: Remove from rdy_nodes & Add to out_nodes
        for fn in free_nodes:
            rdy_nodes.remove(fn)
            out_nodes.append(fn)

        #2. Newly ready nodes: Remove from in_nodes & Add to rdy_nodes
        for fn in free_nodes:
            for s in self.graph.successors(fn):
                if set(self.graph.predecessors(s)).issubset(set(out_nodes)):
                    in_nodes.remove(s)
                    rdy_nodes.append(s)                                                                         

        return in_nodes, rdy_nodes, out_nodes

    def print_status(self, clk, in_nodes, rdy_nodes, out_nodes):
        if self.debug:
            print(clk)
            print(">> in_nodes: %s\n   rdy_nodes: %s\n   out_nodes: %s"
                    %(in_nodes, rdy_nodes, out_nodes))
            for key in self.slot.keys():
                print("%s: %s"%(key, self.slot[key]))
            print("#"*20)
