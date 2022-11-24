import networkx as nx
import math

from tvm_layer import *
from tvm_parse import *
from tvm_config import *

class VectorProcessor():
    def __init__(self, model, debug):
        if not model in ['x220', 'x330']:
            raise NotImplementedError("Currently Not Supported Model: %s"%(model))
        self.model = model
        self.slot = {key:[] for key in self.import_slot(model)}
        self.latency = self.import_latency(model)
        self.type = self.import_type(model)
        self.debug = debug

    def import_slot(self, model):
        return Config[model]['Slot']

    def import_latency(self, model):
        return Config[model]['Latency']

    def import_type(self, model):
        return Config[model]['Type']

    ### Primary Functions
    def generate_graph(self, layer, layout, data, kernel, stride):
        """
        Generate Graph
          1. gen_module(): layer configurations --> Relay IR module --> nested TIR
          2. visit_stmts(): nested TIR --> empty graph
          3. color_graph(): empty graph --> colored graph
        """
        if self.debug: print('> Generate Graph')
        nestedTIR = gen_module(self.model, layer, layout, data, kernel, stride) # Relay IR module
        if self.debug: print(nestedTIR)
        self.graph = visit_stmts(nestedTIR, self.debug) # empty graph
        self.color_graph()
    
    def optimize_graph(self):
        """
        Optimize Graph
          1. loop_hoisting(): hoist loop variables & remove scalar nodes
          2. remove_tensor_init_nodes(): remove tensor initialization nodes
          3. remove_nodes(): remove unncessary nodes
          4. optimize()
        """
        if self.debug: print('> Optimize Graph')
        self.loop_hoisting() # raw_graph
        self.remove_tensor_init_nodes(['Input', 'Filter', 'Multiplier', 'Shifter'])
        self.remove_nodes(attr='type', str='Seq')
        self.remove_nodes(attr='name', str='Multiplier')
        self.remove_nodes(attr='name', str='Shifter')

        self.optimize('register')
        self.optimize('vstore')
        self.optimize('filter load') # For kernels that use filter; e.g, Dwconv
        self.optimize('consecutive max')

    def run(self, swp):
        """
        Run
          1. analyze_graph(): analyze graph to geneate time_stamp
          2. get_estimated_cycles(): cycle = sum(duration*extent)
        """
        if self.debug: print('> Run')
        self.analyze_graph()
        self.estimate_cycles(swp)
        return self.estimated_cycles

    ### Secondary Functions
    def color_graph(self):
        if self.debug: print('>> Color Graph')
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

    def loop_hoisting(self):
        if self.debug: print('>> Loop Hoisting')
        loop_var_nodes = self.get_nodes(attr='type', str='For Start')
        loop_var_names = [self.graph.nodes[node]['var'] for node in loop_var_nodes]

        int_var_nodes = self.get_nodes(attr='type', str='Int Var')
        int_var_names = [self.graph.nodes[node]['var'] for node in int_var_nodes]

        for int_var_name in int_var_names:
            if int_var_name.rsplit('_', 1)[0] == 'cse_var': continue
            loop_var_node = loop_var_nodes[loop_var_names.index(int_var_name)]
            self.graph.nodes[loop_var_node]['scalar_time'] += 1

        self.remove_nodes(attr='slot', str='Scalar')

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
            if self.debug: print('>> Removing Tensor Init Nodes (%s)'%(name))
            node_num = store_nodes[idx]
            index = [idx for idx, sg in enumerate(subgraph) if node_num in sg]
            for idx in index:
                sg = subgraph[idx]
                self.graph.remove_nodes_from(sg)

    def remove_nodes(self, attr, str):
        if any([self.graph.nodes[node][attr]==str for node in self.graph.nodes]):
            if self.debug: print('>> Remove Nodes (%s: %s)'%(attr, str))
        while True:
            nodes = [node for node in self.graph.nodes if self.graph.nodes[node][attr]==str]
            if len(nodes)==0: break

            node = nodes[0]
            parents = list(self.graph.predecessors(node))
            children = list(self.graph.successors(node))
            for pair in [(p,c) for p in parents for c in children]:
                self.graph.add_edge(pair[0], pair[1])
            self.graph.remove_node(node)

    def optimize(self, option):
        FLAG = False
        if option == 'register':
            FLAG = self.optimize_register()
        elif option == 'vstore':
            FLAG = self.optimize_vstore()
        elif option == 'filter load':
            FLAG = self.optimize_filter_load()
        elif option == 'consecutive max':
            FLAG = self.optimize_consecutive_max()
        elif option == 'consecutive load':
            FLAG = self.optimize_consecutive_load()
        else:
            raise TypeError("Currently Not Supported Optimization Option: %s"%(option))
        if (FLAG == True) and self.debug: print('>> Optimize Graph (%s)'%(option))

    def analyze_graph(self):
        """
        Analyze Graph
          [Steps]
            1. Update each slot's state for current CLK
            2. For available slot, occupy it with node in candid_nodes with appropriate type
            3. Update node states (in_nodes, candid_nodes, out_nodes)
            4. Update CLK

          [Nodes Status]
            in_nodes:   Ready: X, Processed: X
            rdy_nodes:  Ready: O, Processed: X
            out_nodes:  Ready: O, Processed: O

          [time_stamp]
            {CLK: {'Scalar': _, 'Memory': _, 'Vector': _, 'loop': _}}
            - Scalar, Memory, Vector: when node is newly occupied at CLK
            - loop: infers which loop is running at CLK
        """
        if self.debug: print('>> Analyze Graph')
        self.time_stamp = {} 
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
            self.loop_stack_update(clk, loop_stack)
            self.print_node_status(clk, in_nodes, rdy_nodes, out_nodes)
            clk = clk+1

        self.print_node_status(clk, in_nodes, rdy_nodes, out_nodes)
        self.single_iteration = clk+1
        self.print_time_stamp()

    def estimate_cycles(self, SWP=True):
        if self.debug: print('>> Estimate Cycles w/ SWP %s'%('ON' if SWP==True else 'OFF'))
        name = ['NO LOOP']
        nodes = ['']
        extent = [1]
        scalar_time = [0]
        duration = [0]
        depth = [0]
        for time in self.time_stamp.keys():
            node_num = self.time_stamp[time]['loop']
            if node_num == 'NO LOOP': # depth = 0
                duration[-1] += 1
            else: # depth > 0
                node = self.graph.nodes[node_num]
                if (node['type']=='For Start'):
                    if depth[-1] != node['depth']:
                        duration.append(0)
                        name.append(node['name'])
                        nodes.append(node_num)
                        if node['depth']==depth[-1]+1: extent.append(extent[-1]*node['extent'])
                        elif node['depth']==depth[-1]-1: extent.append(extent[name.index(node['name'])])
                        else: raise NotImplementedError("Currently Not Supported case!")
                        scalar_time.append(node['scalar_time'])
                        depth.append(node['depth']) # update depth
                    duration[-1] += 1
                else:
                    raise RuntimeError("Currently Not Supported control node type: %s"%(node['type']))

        name.append('NO LOOP')
        nodes.append('')
        extent.append(1)
        scalar_time.append(0)
        duration.append(0)
        depth.append(0)

        if SWP == True:
            local_maxima = [] # SWP-able loops
            for idx, node in enumerate(depth[1:-1]):
                if depth[idx] > depth[idx-1] and depth[idx] > depth[idx+1]:
                    local_maxima.append(idx)
            for lm in local_maxima:
                SI = self.single_iteration
                II = self.get_initiation_interval(lm, nodes, scalar_time, duration)
                CI = math.ceil(SI/II)
                node_num = nodes[lm]
                innermost_iter = (self.graph.nodes[node_num]['extent']-CI+1)*extent[lm-1]

                if self.debug:
                    print("SWP loop: %s"%(name[lm]))
                    print("SI: %s, II: %s, CI: %s, innermost_iter: %s"%(SI, II, CI, innermost_iter))

                prologue = math.floor((SI-1)/II)*II
                epilogue = (SI+(CI-2)*II) - prologue
                kernel = II
                duration[lm] = (prologue, kernel, epilogue)
                extent[lm] = (extent[lm-1], innermost_iter, extent[lm-1])

        if self.debug:
            print("name:        %s"%(name))
            print("nodes:       %s"%(nodes))
            print("depth:       %s"%(depth))
            print("extent:      %s"%(extent))
            print("scalar_time: %s"%(scalar_time))
            print("duration:    %s"%(duration))
        if not all(len(name) == len(l) for l in [name, depth, extent, scalar_time, duration]):
            raise RuntimeError("Error!!: 'name', 'depth', 'extent', 'scalar_time', 'duration' must have same length!!")

        cycles = []
        self.estimated_cycles = 0
        for i in range(0, len(name)):
            cycles.append("%s*max(%s, %s)"%(extent[i], scalar_time[i], duration[i]))
            if isinstance(extent[i], tuple) and isinstance(duration[i], tuple):
                for j in range(0, len(extent[i])):
                    self.estimated_cycles += extent[i][j]*max(scalar_time[i], duration[i][j])
            else:
                self.estimated_cycles += extent[i]*max(scalar_time[i], duration[i])
        if self.debug:
            print(">> Total Cycles: %s <- %s"%(self.estimated_cycles, cycles))

    ### Tertiary Functions
    def get_nodes(self, attr, str):
        """Search node with certain attribute.

        Parameters
        ----------
        attr : string (e.g., 'type', 'name', 'slot', ...)
        str : string or list of string (e.g., 'Store' or ['Store', 'Load'] ...)

        Returns
        -------
        nodes : list of node_num

        """
        if not isinstance(str, list): str = [str]
        return [node for node in self.graph.nodes if self.graph.nodes[node][attr] in str]

    def get_nodes_with_name(self, attr, str, name):
        """Search node with certain attribute and pattern in name.

        Parameters
        ----------
        attr : string
        str : string or list of string
        name : string

        Returns
        -------
        nodes : list of node_num

        """
        if not isinstance(str, list): str = [str]
        nodes = [node for node in self.graph.nodes if self.graph.nodes[node][attr] in str]
        nodes = [node for node in nodes if name in self.graph.nodes[node]['name']]
        names = sorted(list(dict.fromkeys([self.graph.nodes[node]['name'] for node in nodes])))
        node_groups = []
        for name in names: # Group nodes with identical name
            node_groups.append([node for node in nodes if self.graph.nodes[node]['name']==name])
        return node_groups

    def get_subgraph(self, sorted_nodes):
        # get subgraph of depth=0
        loop_stack = []
        tmp = []
        depth = 0
        seq_flag = False
        for node in sorted_nodes:
            if self.graph.nodes[node]['type'] == 'For Start':
                depth += 1
            if self.graph.nodes[node]['type'] == 'Seq' and depth == 0 and seq_flag == True:
                seq_flag = False
                loop_stack.append(tmp)
                tmp = []
            tmp.append(node)
            if self.graph.nodes[node]['type'] == 'Seq' and depth == 0 and seq_flag == False:
                seq_flag = True
                continue
            if self.graph.nodes[node]['type'] == 'For End':
                depth -= 1
                if depth == 0:
                    loop_stack.append(tmp)
                    tmp = []
        return loop_stack

    def slot_update(self, clk):
        free_nodes = []
        #1. Free slot w/ finished op & Add finished op to 'free_nodes'
        for s in self.slot.keys():
            if not self.slot[s]==[]:
                remove_nodes = []
                for node in self.slot[s]:
                    if node['end']==clk:
                        free_nodes.append(node['node'])
                        remove_nodes.append(node)
                [self.slot[s].remove(rm_node) for rm_node in remove_nodes]

        #2. Remove duplicates in list (; All three slots are occupied by 'Control')
        free_nodes = list(dict.fromkeys(free_nodes))
        return free_nodes

    def slot_occupy(self, rdy_nodes, clk):
        """
        Occupy slot
          1. If 'Control' slot, occupy when all slots are IDLE.
          2. If NOT 'Control' slot, occupy when corresponding slot was NOT occupied during current clk
        """
        for rn in rdy_nodes:
            slot = self.graph.nodes[rn]['slot']
            type = self.graph.nodes[rn]['type']
            optimize = self.graph.nodes[rn]['optimize'] if 'optimize' in self.graph.nodes[rn].keys() else 0
            try:
                dur = self.latency[slot][type][optimize]
            except:
                raise TypeError("Currently Not Supported latency config; Slot: %s, Type: %s"%(slot, type))
            if slot == 'Control':
                if all([self.slot[s]==[] for s in self.slot.keys()]): # If all slots are IDLE, 'Occupy'!!
                    for s in self.slot.keys():
                        self.slot[s].append({'node': rn, 'start': clk, 'end': clk+dur}) # Occupy
                        # self.time_stamp_update(clk, 'Control', rn)
                else: # Else, move to next clk
                    continue
            else:
                CHECK_START_TIME = all([slot_node['start']!=clk for slot_node in self.slot[slot]])
                CHECK_SAME_NODE = all([slot_node['node']!=rn for slot_node in self.slot[slot]])
                if CHECK_START_TIME and CHECK_SAME_NODE:
                    self.slot[slot].append({'node': rn, 'start': clk, 'end': clk+dur}) # Occupy
                    self.time_stamp_update(clk, slot, rn)

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
                    tmp.append(s)
        tmp = list(dict.fromkeys(tmp)) # remove duplicates
        in_nodes = [n for n in in_nodes if n not in tmp] # remove newly_ready_nodes from in_nodes
        rdy_nodes.extend(tmp) # add newly_ready_nodes to rdy_nodes
        return in_nodes, rdy_nodes, out_nodes

    def time_stamp_update(self, clk, str, node_num):
        if clk not in self.time_stamp.keys():
            self.time_stamp[clk] = {}
        self.time_stamp[clk][str] = node_num

    def loop_stack_update(self, clk, loop_stack):
        occupied_nodes = self.get_occupied_nodes()
        if len(occupied_nodes)==1 and self.graph.nodes[occupied_nodes[0]]['slot']=='Control':
            if self.graph.nodes[occupied_nodes[0]]['type']=='For Start':
                loop_stack.append(occupied_nodes[0])
                self.time_stamp_update(clk, 'loop', loop_stack[-1])
            elif self.graph.nodes[occupied_nodes[0]]['type']=='For End':
                self.time_stamp_update(clk, 'loop', loop_stack[-1])
                loop_stack.pop()
        else:
            self.time_stamp_update(clk, 'loop', loop_stack[-1])

    def get_occupied_nodes(self):
        occupied_nodes = []
        [occupied_nodes.extend(self.slot[s]) for s in self.slot.keys() if self.slot[s] is not None]
        occupied_nodes = [node['node'] for node in occupied_nodes]
        return list(dict.fromkeys(occupied_nodes)) # remove duplicates in list

    def optimize_register(self):
        """ Register is used to save temporary data, ommiting cache access """
        FLAG = False
        register_nodes = self.get_nodes_with_name(attr='type', str=['Store', 'Load'], name='Reg')
        if len(register_nodes) > 0: FLAG=True
        for register in register_nodes:
            for node in register:
                parents = list(self.graph.predecessors(node))
                children = list(self.graph.successors(node))
                for t in ((p, c) for p in parents for c in children):
                    self.graph.add_edge(*t)
                self.graph.remove_node(node)
        # for register in register_nodes:
        #     store_nodes = [node for node in register if self.graph.nodes[node]['type']=='Store']
        #     load_nodes = [node for node in register if self.graph.nodes[node]['type']=='Load']
        #     CONNECTED = False
        #     for t in ((s, l) for s in store_nodes for l in load_nodes):
        #         if self.graph.has_edge(*t): # t[0]: store_node, t[1]: load_node
        #             CONNECTED = True
        #             self.graph.remove_edge(t[0], t[1])
        #             # Connect store_node's parent --> store_node's children
        #             grand_parents = [node for node in self.graph.predecessors(t[0])]
        #             children = [node for node in self.graph.successors(t[0])]
        #             for t in ((gp, c) for gp in grand_parents for c in children):
        #                 self.graph.add_edge(*t)
        #             # Connect load_node's parent --> load_node's children
        #             parents = [node for node in self.graph.predecessors(t[1])]
        #             grand_children = [node for node in self.graph.successors(t[1])]
        #             for t in ((p, gc) for p in parents for gc in grand_children):
        #                 self.graph.add_edge(*t)
        #     if CONNECTED==True: [self.graph.remove_node(node) for node in register]
        return FLAG

    def optimize_vstore(self):
        """ Vstore at the end of loop can be bypassed """
        FLAG = False
        store_nodes = self.get_nodes(attr='type', str='Store')
        for node in store_nodes:
            children = self.graph.successors(node)
            if all(self.graph.nodes[child]['type']=='For End' for child in children):
                FLAG = True
                self.graph.nodes[node]['optimize'] = 1
        return FLAG

    def optimize_filter_load(self):
        """ Filter is loaded from Scalar Memory """
        # (Parent)                     (Parent)
        #    |                         /      \
        # (Vector LD)          (Scalar LD)  (Scalar LD)
        #    |  6 cycle                \      /  1 cycle
        # (Child)               (Vector Ch-wise Concat)
        #                                  |  1 cycle
        #                               (Child)
        FLAG = False
        load_nodes = [node for node in self.graph.nodes if self.graph.nodes[node]['type']=='Load' and \
                                                            self.graph.nodes[node]['name']=='Filter']
        if len(load_nodes)==0: return FLAG
        else:
            FLAG = True
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
                        if key != 'name': self.graph.nodes[new_node][key] = self.graph.nodes[node][key]
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
            return FLAG

    def optimize_consecutive_max(self):
        """ Consecutive MAX can be bypassed (MAXP_3x3) """
        FLAG = False
        max_nodes = self.get_nodes(attr='type', str='Max')
        for node in max_nodes:
            children = self.graph.successors(node)
            if all(self.graph.nodes[child]['type']=='Max' for child in children):
                FLAG = True
                self.graph.nodes[node]['optimize'] = 1
        return FLAG

    def print_node_status(self, clk, in_nodes, rdy_nodes, out_nodes):
        if self.debug:
            print("CLK: %s"%(clk))
            print(">> in_nodes: %s\n   rdy_nodes: %s\n   out_nodes: %s"
                    %(in_nodes, rdy_nodes, out_nodes))
            for key in self.slot.keys():
                print("%s: %s"%(key, self.slot[key]))
            print("#"*20)

    def print_time_stamp(self):
        """
        time_stamp: {CLK: {'Scalar': A, 'Memory': B, 'Vector': C, 'loop': D}}
          - Scalar, Memory, Vector: when node is newly occupied at CLK
          - loop: infers which loop is running at CLK
        """
        if self.debug:
            for time in self.time_stamp.keys():
                node_num = self.time_stamp[time]['loop']
                print("#(%d): %s --> %s"%(time, self.time_stamp[time], 'X' if node_num=='NO LOOP' else self.graph.nodes[node_num]))
                # node_num, info = self.time_stamp[time]['loop'], self.graph.nodes[self.time_stamp[time]['loop']]
                # print("#(%d): %d, %s"%(time, node_num, info))

    def get_initiation_interval(self, idx, nodes, scalar_time, duration):
        if self.debug: print(">>> Get Initiation Interval")
        start_time, end_time = sum(duration[0:idx]), sum(duration[0:idx+1])
        Scalar, Memory, Vector = scalar_time[idx], 0, 0
        max_instr_latency = 0
        for time in range(start_time, end_time):
            if 'Memory' in self.time_stamp[time].keys():
                Memory+=1
                node_num = self.time_stamp[time]['Memory']
                type = self.graph.nodes[node_num]['type']
                optimize = self.graph.nodes[node_num]['optimize'] if 'optimize' in self.graph.nodes[node_num].keys() else 0
                instr_latency = self.latency['Memory'][type][optimize]
                if any(self.graph.nodes[child]['type']=='Memory' for child in self.graph.successors(node_num)):
                    instr_latency+=1
                max_instr_latency = max(max_instr_latency, instr_latency)
            if 'Vector' in self.time_stamp[time].keys():
                Vector+=1
                node_num = self.time_stamp[time]['Vector']
                type = self.graph.nodes[node_num]['type']
                optimize = self.graph.nodes[node_num]['optimize'] if 'optimize' in self.graph.nodes[node_num].keys() else 0
                instr_latency = self.latency['Vector'][type][optimize]
                if any(self.graph.nodes[child]['type']=='Vector' for child in self.graph.successors(node_num)):
                    instr_latency+=1
                max_instr_latency = max(max_instr_latency, instr_latency)
        min_length = max(Scalar, Memory, Vector)
        if self.debug:
            print("    (Scalar, Memory, Vector): (%s, %s, %s)"%(Scalar, Memory, Vector))
            print("    max_instr_latency: %s"%(max_instr_latency))
        initiation_interval = max(min_length, max_instr_latency)
        return initiation_interval
