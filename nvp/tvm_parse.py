from ast import Num
from inspect import Attribute
import tvm
from tvm.tir import expr as _expr
from tvm.tir import stmt as _stmt
from tvm.ir.op import Op as _Op

import networkx as nx
import re

from tvm_visualize import *

# FIXME: remove SEQ_DEBUG
SEQ_DEBUG=0

def Dprint(message, DEBUG=False, end='\n'):
    if DEBUG: print(message, end=end)

def visit_stmts(primfn, DEBUG):
    g = nx.DiGraph()
    g = generate_nodes(primfn.body, g, DEPTH=0, WS="", DEBUG=DEBUG)
    if not nx.is_directed_acyclic_graph(g):
        print("Currenlty Not Supported: Graph must be DAG!!")
        print(nx.find_cycle(g, orientation="original"))
        # assert 0
    return g

def generate_nodes(stmt, G, DEPTH, WS="", DEBUG=False): # tir.stmt.___
    if isinstance(stmt, (_stmt.For)):
        DEPTH = DEPTH+1
        g = nx.DiGraph()
        Dprint(WS+' '+str(type(stmt))+' Start(%d)'%(DEPTH), DEBUG)
        g = generate_nodes(stmt.body, g, DEPTH, WS+'|', DEBUG)
        Dprint(WS+' '+str(type(stmt))+' End(%d)'%(DEPTH), DEBUG)

        start = max(list(g.nodes))+1
        end = start+1
        VAR = stmt.loop_var.name
        g.add_node(start, name='For %s[%d]'%(VAR, DEPTH), type='For Start', depth=DEPTH, var=VAR, extent=stmt.extent, scalar_time=0)
        [g.add_edge(start, node) for node in g.nodes if g.in_degree(node)==0 and not node==start] # Connect 'For Start' with 'Src'
        
        g.add_node(end, name='For %s[%d]'%(VAR, DEPTH), type='For End', depth=DEPTH, var=VAR)
        [g.add_edge(node, end) for node in g.nodes if g.out_degree(node)==0 and not node==end] # Connect 'For End' with 'Sink'
        DEPTH = DEPTH-1
        return g

    elif isinstance(stmt, _stmt.SeqStmt):
        Dprint(WS+" "+str(type(stmt)), DEBUG)
        glist = []
        seq_group = list(range(0, len(stmt.seq)))

        for idx, st in enumerate(stmt.seq):
            g = nx.DiGraph()
            g = generate_nodes(st, g, DEPTH, WS+str(idx), DEBUG)
            glist.append(g)
            # acc_node_cnt.append(len(glist[-1].nodes)+acc_node_cnt[idx])
            if (type(st)==_stmt.For): seq_group[idx] = [seq_group[idx]]

        # seq_group is a double nested list: e.g., [[1, 2], [3], [4, 5, 6]]
        # Dependency between depth-0 lists in seq_group: e.g., [1, 2] <-> [3] <-> [4, 5, 6]
        # NO dependency betwen elements in depth-1 list: e.g., [1, 2] == [2, 1]
        seq_group = group_seqstmts(seq_group)

        g_union_list = []
        acc_node_cnt = [0] # accumulated node counts
        for idx, group in enumerate(seq_group):
            ## Add RAW dependency within a seq_group
            sub_acc_node_cnt = [0]
            sub_g_union_list = []
            for gidx, g in enumerate(group):
                sub_g_union_list.append(glist[g])
                sub_acc_node_cnt.append(len(sub_g_union_list[-1].nodes)+sub_acc_node_cnt[gidx])
            sub_g_union = nx.disjoint_union_all(sub_g_union_list)

            ## Calculate Reaching Definition (Store -> Load)
            ## > Iterate over sub_g_union 'Backwards'
            ## > Check if following there exists a path between following store-nodes and the load-node
            ## > Definition reaches only if paths DON'T exist!!
            for sidx0 in range(len(group)-1, -1, -1): # sub index
                store_nodes0 = [node for node in list(sub_g_union)[sub_acc_node_cnt[sidx0]:sub_acc_node_cnt[sidx0+1]] if sub_g_union.nodes[node]['type']=='Store']
                for store_node in store_nodes0:
                    store_name = sub_g_union.nodes[store_node]['name']
                    next_store_nodes = [node for node in list(sub_g_union)[sub_acc_node_cnt[sidx0+1]:] if sub_g_union.nodes[node]['type']=='Store' and sub_g_union.nodes[node]['name']==store_name]
                    for sidx1 in range(sidx0+1, len(group)): # sub index
                        load_nodes = [node for node in list(sub_g_union)[sub_acc_node_cnt[sidx1]:sub_acc_node_cnt[sidx1+1]]
                                        if sub_g_union.nodes[node]['type']=='Load' and sub_g_union.nodes[node]['name']==store_name]
                        for load_node in load_nodes:
                            # reaching_def = not any(nx.has_path(sub_g_union, store_node, next_store_node) for next_store_node in next_store_nodes)
                            reaching_def = not any(nx.has_path(sub_g_union, next_store_node, load_node) for next_store_node in next_store_nodes)
                            has_path = [nx.has_path(sub_g_union, next_store_node, load_node) for next_store_node in next_store_nodes]
                            # print(store_node, store_name, load_node, reaching_def, next_store_nodes, has_path)
                            if reaching_def:
                                sub_g_union.add_edge(store_node, load_node)
                            else:
                                pass
            ##
            g_union_list.append(sub_g_union)
            acc_node_cnt.append(len(g_union_list[-1].nodes)+acc_node_cnt[idx])
        g_union = nx.disjoint_union_all(g_union_list)

        ## Connect neighboring seq_group with Seq node; n-th group sinks --> Seq --> (n+1)-th group srces (n=0,...,N-1)
        for idx in range(1, len(acc_node_cnt)-1):
            node_num = len(g_union.nodes)
            g_union.add_node(node_num, name='Seq', type='Seq')
            # Connect N-th group sinks --> Seq --> (N+1)-th group srces
            sinks = [node for node in list(g_union.nodes)[acc_node_cnt[idx-1]:acc_node_cnt[idx]] if g_union.out_degree(node)==0]
            [g_union.add_edge(sink, node_num) for sink in sinks]
            srces = [node for node in list(g_union.nodes)[acc_node_cnt[idx]:acc_node_cnt[idx+1]] if g_union.in_degree(node)==0]
            [g_union.add_edge(node_num, src) for src in srces]

        return g_union

    elif isinstance(stmt, _stmt.Allocate):
        Dprint(WS+" "+str(type(stmt)), DEBUG)
        G = generate_nodes(stmt.body, G, DEPTH, WS, DEBUG)
        return G

    else: # Generate Data Nodes
        g = nx.DiGraph()
        g.add_node(0, name="DUMMY")
        g = generate_data_nodes(stmt, g, 0, WS, DEBUG)
        g.remove_node(0)
        return g

def generate_data_nodes(stmt, g, parent, WS, DEBUG):
    if isinstance(stmt, _stmt.BufferStore):
        g = visit_STORE(stmt, g, parent, WS, DEBUG)
    elif isinstance(stmt, _stmt.LetStmt):
        g = visit_LET(stmt, g, parent, WS, DEBUG)
    elif isinstance(stmt, tvm.ir.PrimExpr):
        g = visit_EXPR(stmt, g, parent, WS, DEBUG)
    else:
        raise NotImplementedError("Not Implemented Error: %s"%(str(type(stmt))))
    return g

def visit_STORE(stmt, g, parent, WS, DEBUG):
    node_num = len(g.nodes)
    Dprint(WS+" "+str(type(stmt))+"\t[Memory](%d)"%(node_num), DEBUG)
    Dprint(WS+" ##################### "+"@Location to be stored"+" #####################", DEBUG)
    g.add_node(node_num, name=str(stmt).split('=')[0].split('[')[0], type='Store')
    g.add_edge(node_num, parent)
    for idx in range(0, len(stmt.indices)):
        g = generate_data_nodes(stmt.indices[idx], g, node_num, WS+str(idx), DEBUG)
    Dprint(WS+" ##################### "+"@Value to be stored"+" #####################", DEBUG)
    g = generate_data_nodes(stmt.value, g, node_num, WS+'|', DEBUG)
    return g

def visit_LET(stmt, g, parent, WS, DEBUG):
    g = generate_data_nodes(stmt.var, g, parent, WS, DEBUG) # cse_var_1: int32
    var_num = len(g.nodes)-1
    g = generate_data_nodes(stmt.value, g, len(g.nodes)-1, WS, DEBUG) # (((i1: int32*25) + (i2: int32*5)) + i3: int32)
    # Remove edge connection with parent node before graph merge
    [g.remove_edge(node_num, parent) for node_num in g.nodes if g.has_edge(node_num, parent)]

    ## Make temporary DiGraph to merge w/ the original one
    g_ = nx.DiGraph()
    g_ = generate_nodes(stmt.body, g_, 0, WS, DEBUG) # pad_temp[cse_var_1] = placeholder[cse_var_1]
    # seq_nodes = [node for node in g_.nodes if g_.nodes[node]['type']=='Seq']
    # for seq_node in seq_nodes: # remove edge: seq_node --> cse_var
    #     children = list(g_.successors(seq_node))
    #     [g_.remove_edge(seq_node, child) for child in children if g_.nodes[child]['name']==g.nodes[var_num]['name']]

    ## Merge two graphs with new node_num
    u = nx.disjoint_union(g, g_)

    ## Merge nodes that have same name as 'var_num'
    merge_num = [num for num in u.nodes if (u.nodes[num]['name']==u.nodes[var_num]['name']) and (num is not var_num)]
    for n in merge_num:
        u = nx.contracted_nodes(u, var_num, n)
    [u.add_edge(sink, parent) for sink in u.nodes if u.out_degree(sink)==0 and not sink==0] # sink==0: KERNEL_END node
    return u

def visit_EXPR(expr, g, parent, WS, DEBUG): # tir.expr.__
    if hasattr(expr, "indices"): # BufferLoad
        node_num = len(g.nodes)
        Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        g.add_node(node_num, name=str(expr).split(':')[0], type='Load')
        g.add_edge(node_num, parent)
        for idx in range(0, len(expr.indices)):
            g = visit_EXPR(expr.indices[idx], g, node_num, WS+str(idx), DEBUG)
    elif all(hasattr(expr, attr) for attr in ["a", "b"]): # Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, EA, NE, LT, LE, GT, GE, And, Or
        node_num = len(g.nodes)
        Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        try:
            found = re.search('<class \'tvm.tir.expr.(.+?)\'>', str(type(expr))).group(1)
        except AttributeError:
            print("%s NOT supported" %(type(expr)))
            assert 0
        g.add_node(node_num, name=str(expr), type=found)
        g.add_edge(node_num, parent)
        g = visit_EXPR(expr.a, g, node_num, WS+"L", DEBUG)
        g = visit_EXPR(expr.b, g, node_num, WS+"R", DEBUG)
    elif isinstance(expr, _expr.Cast): # Cast
        node_num = len(g.nodes)
        Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        g = visit_EXPR(expr.value, g, parent, WS+"*", DEBUG)
    elif isinstance(expr, _expr.Call): # Call
        node_num = len(g.nodes)
        Dprint(WS+" "+str(type(expr.op))+" --> "+str(expr.op), DEBUG)
        if hasattr(expr, "op"):
            if expr.op in [_Op.get("tir.shift_left"), _Op.get("tir.shift_right")]:
                g.add_node(node_num, name=str(expr), type='Bit Shift')
                g.add_edge(node_num, parent)
            elif expr.op in [_Op.get("tir.sigmoid")]:
                g.add_node(node_num, name=str(expr), type='LUT')
                g.add_edge(node_num, parent)
            else:
                raise NotImplementedError("Currently NOT supported expr.Call.op type: %s"%(expr.op))
        else:
            raise NotImplementedError("Currently NOT supported expr.Call type: %s"%(expr))
        for idx in range(0, len(expr.args)):
            g = visit_EXPR(expr.args[idx], g, node_num, WS+str(idx), DEBUG)
    elif isinstance(expr, (_expr.Var)): # Var
        node_num = len(g.nodes)
        if node_num == parent: assert 0, str(expr)
        if expr.dtype == 'int32':
            Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
            g.add_node(node_num, name=str(expr), type='Int Var', var=expr.name)
            g.add_edge(node_num, parent)
        elif expr.dtype == 'float32':
            Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
            g.add_node(node_num, name=str(expr), type='Float Var', var=expr.name)
            g.add_edge(node_num, parent)
        else:
            raise NotImplementedError("Currently NOT supported variable type: %s"%(expr.dtype))
    elif isinstance(expr, (_expr.IntImm)): # IntImm
        node_num = len(g.nodes)
        Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        g.add_node(node_num, name=str(expr), type='Int Imm')
        g.add_edge(node_num, parent)
    elif isinstance(expr, (_expr.FloatImm)): # FloatImm
        node_num = len(g.nodes)
        Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        g.add_node(node_num, name=str(expr), type='Float Imm')
        g.add_edge(node_num, parent)
    else:
        Dprint("##"*10, DEBUG)
        Dprint(type(expr), DEBUG)
        Dprint(str(expr), DEBUG)
        Dprint(str(expr.name), DEBUG)
        Dprint("Currently Not Supported!!!!!", DEBUG)
        assert 0
    return g

def group_seqstmts(seq_group):
    # Group SeqStmts; No bypassing across For-loop
    seq_group_ = []
    seq_temp = []
    for idx in range(0, len(seq_group)):
        if isinstance(seq_group[idx], list):
            if not seq_temp==[]: seq_group_.append(seq_temp)
            seq_group_.append(seq_group[idx])
            seq_temp = []
        else:
            seq_temp.append(seq_group[idx])
    if not seq_temp==[]: seq_group_.append(seq_temp)
    return seq_group_