from ast import Num
from inspect import Attribute
import tvm
from tvm.tir import expr as _expr
from tvm.tir import stmt as _stmt

import networkx as nx
import re

from tvm_visualize import *

# FIXME: remove SEQ_DEBUG
SEQ_DEBUG=0

def Dprint(message, DEBUG=False):
    if DEBUG: print(message)

def visit_stmts(primfn, DEBUG):
    g = nx.DiGraph()
    g = generate_nodes(primfn.body, g, DEPTH=0, WS="", DEBUG=DEBUG)
    return g

def generate_nodes(stmt, G, DEPTH, WS="", DEBUG=False): # tir.stmt.___
    if isinstance(stmt, (_stmt.For)):
        DEPTH = DEPTH+1
        g = nx.DiGraph()
        Dprint(WS+' '+str(type(stmt))+' Start(%d)'%(DEPTH), DEBUG)
        g = generate_nodes(stmt.body, g, DEPTH, WS+'|', DEBUG)
        Dprint(WS+' '+str(type(stmt))+' End(%d)'%(DEPTH), DEBUG)

        # if all([g.nodes[node]['type'] != 'Seq' for node in g.nodes]): # if there is no 'Seq' node in the subgraph
        #     g, bool = merge_redundant_nodes(g)
        #     if bool: save_graph_viz(g, 'typeNum', 'graph_reduced%d.png'%(DEPTH))

        start = max(list(g.nodes))+1
        end = start+1
        VAR = stmt.loop_var.name
        g.add_node(start, name='For %s[%d]'%(VAR, DEPTH), type='For Start', depth=DEPTH, var=VAR, extent=stmt.extent, min_time=0)
        [g.add_edge(start, node) for node in g.nodes if g.in_degree(node)==0 and not node==start] # Connect 'For Start' with 'Src'
        
        g.add_node(end, name='For %s[%d]'%(VAR, DEPTH), type='For End', depth=DEPTH, var=VAR)
        [g.add_edge(node, end) for node in g.nodes if g.out_degree(node)==0 and not node==end] # Connect 'For End' with 'Sink'
        DEPTH = DEPTH-1
        return g

    elif isinstance(stmt, _stmt.SeqStmt):
        Dprint(WS+" "+str(type(stmt)), DEBUG)
        glist = []
        acc_node_cnt = [0] # accumulated node counts of graph in glist
        for idx, st in enumerate(stmt.seq):
            g = nx.DiGraph()
            g = generate_nodes(st, g, DEPTH, WS+str(idx), DEBUG)

            if DEBUG:
                ## SEQ_DEBUG
                global SEQ_DEBUG
                save_graph_viz(g, 'typeNum', 'graph%d.png'%(SEQ_DEBUG))
                SEQ_DEBUG=SEQ_DEBUG+1
                ## SEQ_DEBUG

            # if all([g.nodes[node]['type'] != 'Seq' for node in g.nodes]): # if there is no 'Seq' node in the subgraph
            #     g, bool = merge_redundant_nodes(g)
            #     if bool: save_graph_viz(g, 'typeNum', 'graph%d_reduced.png'%(SEQ_DEBUG-1))

            glist.append(g)
            acc_node_cnt.append(len(glist[-1].nodes)+acc_node_cnt[idx])
        g_union = nx.disjoint_union_all(glist)

        ##Connect glist[N]::sink <-> Seq <-> glist[N+1]::src
        sinks, srces = [], []
        for idx in range(0, len(acc_node_cnt)-2):
            sinks = [node for node in list(g_union.nodes)[acc_node_cnt[idx]:acc_node_cnt[idx+1]] if g_union.out_degree(node)==0]
            srces = [node for node in list(g_union.nodes)[acc_node_cnt[idx+1]:acc_node_cnt[idx+2]] if g_union.in_degree(node)==0]
            node_num = len(g_union.nodes)
            g_union.add_node(node_num, name='Seq', type='Seq')
            [g_union.add_edge(sink, node_num) for sink in sinks]
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
    if stmt.buffer.name == 'Acc':
        node_num = len(g.nodes)-1
    else:
        node_num = len(g.nodes)
        Dprint(WS+" "+str(type(stmt))+"\t[Memory](%d)"%(node_num), DEBUG)
        Dprint(WS+" ##################### "+"@Location to be stored"+" #####################", DEBUG)
        g.add_node(node_num, name=str(stmt), type='Store')
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
        if expr.buffer.name == 'Acc':
            node_num = len(g.nodes)-1
        else:
            node_num = len(g.nodes)
            Dprint(WS+" "+str(type(expr))+" --> "+str(expr), DEBUG)
            g.add_node(node_num, name=str(expr), type='Load')
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
        Dprint(WS+" "+str(type(expr.op))+" --> "+str(expr.op), DEBUG)
        for idx in range(0, len(expr.args)):
            g = visit_EXPR(expr.args[idx], g, parent, WS+str(idx), DEBUG)
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

def merge_redundant_nodes(g):
    nodes = g.nodes
    names = [g.nodes[node]['name'] for node in nodes]
    redundant_names = set([n for n in names if names.count(n)>1])
    if len(redundant_names)==0:
        return g, False

    print("########## Removing Redundant Nodes ##########")
    [print(rdd_name) for rdd_name in redundant_names]

    rdd_groups = [] # [[nodes with name 'A'], [nodes with name 'B'], ...]
    for rdd_name in redundant_names:
        rdd_groups.append([node for node in g.nodes if g.nodes[node]['name']==rdd_name])
 
    for rdd_group in rdd_groups:
        for rdd in rdd_group[1:]: # Merge all redundant nodes to a single node
            g = nx.contracted_nodes(g, rdd_group[0], rdd)

    return g, True
