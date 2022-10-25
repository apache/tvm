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
        self.time_stamp = {}

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

    def loop_hoisting(self):
        loop_var_nodes = self.get_nodes(attr='type', str='For Start')
        loop_var_names = [self.graph.nodes[node]['var'] for node in loop_var_nodes]

        int_var_nodes = self.get_nodes(attr='type', str='Int Var')
        int_var_names = [self.graph.nodes[node]['var'] for node in int_var_nodes]

        for int_var_name in int_var_names:
            if int_var_name.rsplit('_', 1)[0] == 'cse_var': continue
            loop_var_node = loop_var_nodes[loop_var_names.index(int_var_name)]
            self.graph.nodes[loop_var_node]['min_time'] += 1

        self.remove_nodes(attr='slot', str='Scalar')
        return

    def remove_tensor_init_nodes(self, names):
        sorted_nodes = list(nx.topological_sort(self.graph))
        subgraph = self.get_subgraph(sorted_nodes)
        store_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['type']=='Store']
        # 'Input[cse_var_1] = p0[cse_var_1]\n' --> 'Input[cse_var_1]'
        store_names = [self.graph.nodes[node]['name'].split('=')[0].split('[')[0].strip() for node in store_nodes]
        for name in names:
            try:
                idx = store_names.index(name)
            except:
                continue
            print('########## Removing Tensor Init Nodes (%s) ##########'%(name))
            node_num = store_nodes[idx]
            index = [idx for idx, sg in enumerate(subgraph) if node_num in sg]
            for idx in index:
                sg = subgraph[idx]
                self.graph.remove_nodes_from(sg)
        return

    def get_subgraph(self, sorted_nodes):
        loop_stack = []
        tmp = []
        depth = 0
        for node in sorted_nodes:
            if self.graph.nodes[node]['type'] == 'For Start':
                depth += 1
            tmp.append(node)
            if self.graph.nodes[node]['type'] == 'For End':
                depth -= 1
                if depth == 0:
                    loop_stack.append(tmp)
                    tmp = []
        return loop_stack

    def optimize(self, option):
        FLAG = False
        if option == 'vstore':
            FLAG = self.optimize_vstore()
        elif option == 'filter load':
            FLAG = self.optimize_filter_load()
        else:
            raise TypeError("Currently Not Supported Optimization Option: %s"%(option))
        if FLAG == True: print('########## Optimize Graph (%s) ##########'%(option))

    def optimize_vstore(self):
        FLAG = False
        store_nodes = self.get_nodes(attr='type', str='Store')
        for node in store_nodes:
            children = self.graph.successors(node)
            if all(self.graph.nodes[child]['type']=='For End' for child in children):
                FLAG = True
                self.graph.nodes[node]['optimize'] = 1
        return FLAG

    def optimize_filter_load(self):
        # (Parent)                     (Parent)
        #    |                         /      \
        # (Vector LD)          (Scalar LD)  (Scalar LD)
        #    |  6 cycle                \      /  1 cycle
        # (Child)               (Vector Ch-wise Concat)
        #                                  |  1 cycle
        #                               (Child)
        load_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['type']=='Load' and \
                                                            self.graph.nodes[node]['name']=='Filter']
        if len(load_nodes)==0: return False
        else:
            for node in load_nodes:
                parents = list(self.graph.predecessors(node))
                children= list(self.graph.successors(node))
                name = str(self.graph.nodes[node]['name'])
                node_num = max(self.graph.nodes)+1

                # Add two 'Scalar LD' nodes
                for e in range(0,2):
                    new_node = node_num + e
                    self.graph.add_node(new_node)
                    self.graph.nodes[new_node]['name'] = name+str(e)
                    for key in self.graph.nodes[node].keys():
                        if key is not 'name': self.graph.nodes[new_node][key] = self.graph.nodes[node][key]
                    self.graph.nodes[new_node]['optimize'] = 1
                    for parent in parents:
                        self.graph.add_edge(parent, new_node)

                # Add 'Vector Ch-wise Concat' node
                new_node = node_num+2
                self.graph.add_node(new_node)
                self.graph.nodes[new_node]['name'] = name+'0'+'::'+name+'1'
                self.graph.nodes[new_node]['type'] = 'Ch Concat'
                self.graph.nodes[new_node]['slot'] = 'Vector'
                self.graph.add_edge(node_num+0, new_node)
                self.graph.add_edge(node_num+1, new_node)
                for child in children:
                    self.graph.add_edge(new_node, child)

                self.graph.remove_node(node)
            return True

    def get_nodes(self, attr, str):
        return [node for node in self.graph.nodes if self.graph.nodes[node][attr]==str]

    def remove_nodes(self, attr, str):
        while True:
            nodes = [node for node in self.graph.nodes if self.graph.nodes[node][attr]==str]
            if len(nodes)==0: break

            node = nodes[0]
            parents = list(self.graph.predecessors(node))
            children = list(self.graph.successors(node))
            for pair in [(p,c) for p in parents for c in children]:
                self.graph.add_edge(pair[0], pair[1])
            self.graph.remove_node(node)
        return

    def run_single_iteration(self):
        '''    
        [Steps in 'run_single_iteration()']
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

        print('########## Run Single Iteration ##########')
        in_nodes, rdy_nodes, out_nodes = [], [], []

        # Initialize: clk=0
        clk = 0
        rdy_nodes = [node for node in self.graph.nodes if self.graph.in_degree(node)==0]
        in_nodes = list(set(self.graph.nodes)-set(rdy_nodes)) # Update in_nodes
        loop_stack = ['NO LOOP']

        while True:
            free_nodes = self.slot_update(clk)
            in_nodes, rdy_nodes, out_nodes = self.node_update(free_nodes, in_nodes, rdy_nodes, out_nodes)
            if len(in_nodes)==0 and len(rdy_nodes)==0:
                break
            self.slot_occupy(rdy_nodes, clk)
            self.time_stamp_update(clk, loop_stack)
            self.print_node_status(clk, in_nodes, rdy_nodes, out_nodes)
            clk = clk+1

        self.print_node_status(clk, in_nodes, rdy_nodes, out_nodes)
        print("Single Iteration Run Time: %d"%(clk))
        self.print_time_stamp_status()

    def get_estimated_cycles(self):
        print('########## Get Estimated Cycles ##########')
        # time_stamp: {time: node_num}
        max_depth = max([self.graph.nodes[node_num]['depth'] for node_num in self.time_stamp.values()])
        name = ['NO LOOP']
        extent = [1]
        min_time = [0]
        duration = [0]*(max_depth+1) # 0-th depth: NO loop area
        for time in self.time_stamp.keys():
            node_num = self.time_stamp[time]
            node = self.graph.nodes[node_num]
            if (node['type']=='For Start'):
                duration[node['depth']] += 1
                if (name[-1]!=node['name']) and (node['name'] not in name):
                    name.append(node['name'])
                    extent.append(extent[-1]*node['extent'])
                    min_time.append(node['min_time'])
            else:
                raise RuntimeError("Currently Not Supported control node type: %s"%(node['type']))

        print("name:     %s"%(name))
        print("extent:   %s"%(extent))
        print("min_time: %s"%(min_time))
        print("duration: %s"%(duration))
        print(name, extent, min_time, duration) # Suhong
        if not all(len(name) == len(l) for l in [name, extent, min_time, duration]):
            raise RuntimeError("Error!!: 'name', 'extent', 'min_time', 'duration' must have same length!!")

        cycles = []
        estimated_cycles = 0
        for i in range(0, len(name)):
            cycles.append("%s*max(%s, %s)"%(extent[i], min_time[i], duration[i]))
            estimated_cycles += extent[i]*max(min_time[i], duration[i])
        print(">> Total Cycles: %s <- %s"%(estimated_cycles, cycles))

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

    def slot_occupy(self, rdy_nodes, clk):
        for rn in rdy_nodes:
            slot = self.graph.nodes[rn]['slot']
            type = self.graph.nodes[rn]['type']
            optimize = self.graph.nodes[rn]['optimize'] if 'optimize' in self.graph.nodes[rn].keys() else 0
            try:
                dur = self.latency[slot][type][optimize]
            except:
                raise TypeError("Currently Not Supported latency config; Slot: %s, Type: %s"%(slot, type))
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

        #2. Newly readied nodes: Remove from in_nodes & Add to rdy_nodes
        tmp = []
        for fn in free_nodes:
            for s in self.graph.successors(fn):
                if set(self.graph.predecessors(s)).issubset(set(out_nodes)):
                    # in_nodes.remove(s)
                    # rdy_nodes.append(s)
                    tmp.append(s)
        tmp = list(dict.fromkeys(tmp)) # remove duplicates
        in_nodes = [n for n in in_nodes if n not in tmp] # remove newly_ready_nodes from in_nodes
        rdy_nodes.extend(tmp) # add newly_ready_nodes to rdy_nodes
        return in_nodes, rdy_nodes, out_nodes

    def time_stamp_update(self, clk, loop_stack):
        occupied_nodes = self.get_occupied_nodes()
        if len(occupied_nodes)==1 and self.graph.nodes[occupied_nodes[0]]['slot']=='Control':
            if self.graph.nodes[occupied_nodes[0]]['type']=='For Start':
                loop_stack.append(occupied_nodes[0])
                self.time_stamp[clk] = loop_stack[-1]
            elif self.graph.nodes[occupied_nodes[0]]['type']=='For End':
                self.time_stamp[clk] = loop_stack[-1]
                loop_stack.pop()
        else:
            # self.time_stamp[clk] = self.time_stamp[clk-1]
            self.time_stamp[clk] = loop_stack[-1]

    def get_occupied_nodes(self):
        occupied_nodes = [self.slot[s]['node'] for s in self.slot.keys() if self.slot[s] is not None]
        return list(dict.fromkeys(occupied_nodes)) # remove duplicates in list

    def print_node_status(self, clk, in_nodes, rdy_nodes, out_nodes):
        if self.debug:
            print("CLK: %s"%(clk))
            print(">> in_nodes: %s\n   rdy_nodes: %s\n   out_nodes: %s"
                    %(in_nodes, rdy_nodes, out_nodes))
            for key in self.slot.keys():
                print("%s: %s"%(key, self.slot[key]))
            print("#"*20)

    def print_time_stamp_status(self):
        if self.debug:
            for time in self.time_stamp.keys():
                node_num, info = self.time_stamp[time], self.graph.nodes[self.time_stamp[time]]
                print("#(%d): %d, %s"%(time, node_num, info))
