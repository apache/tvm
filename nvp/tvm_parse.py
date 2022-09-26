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
    g = generate_nodes(primfn.body, g, depth=0, ws="", DEBUG=DEBUG)
    return g

def generate_nodes(stmt, G, depth, ws="", DEBUG=False): # tir.stmt.___
    if isinstance(stmt, (_stmt.For)):
        depth = depth+1
        g = nx.DiGraph()
        Dprint(ws+' '+str(type(stmt))+' Start(%d)'%(depth), DEBUG)
        g = generate_nodes(stmt.body, g, depth, ws+'|', DEBUG)
        Dprint(ws+' '+str(type(stmt))+' End(%d)'%(depth), DEBUG)

        # Color MISC nodes
        # g = color_misc_nodes(g, DEBUG)

        start = max(list(g.nodes))+1
        end = start+1
        g.add_node(start, name='For Start[%d]'%(depth), type='For Start')
        [g.add_edge(start, node) for node in g.nodes if g.in_degree(node)==0 and not node==start] # Connect 'For Start' with 'Src'
        
        g.add_node(end, name='For End[%d]'%(depth), type='For End')
        [g.add_edge(node, end) for node in g.nodes if g.out_degree(node)==0 and not node==end] # Connect 'For End' with 'Sink'
        depth = depth-1
        return g

    elif isinstance(stmt, _stmt.SeqStmt):
        Dprint(ws+" "+str(type(stmt)), DEBUG)
        glist = []
        acc_node_cnt = [0] # accumulated node counts of graph in glist
        for idx, st in enumerate(stmt.seq):
            g = nx.DiGraph()
            g = generate_nodes(st, g, depth, ws+str(idx), DEBUG)

            # Color MISC nodes
            # g = color_misc_nodes(g, DEBUG)

            if DEBUG:
                ## SEQ_DEBUG
                global SEQ_DEBUG
                save_graph_viz(g, 'typeNum', 'graph%d.png'%(SEQ_DEBUG))
                SEQ_DEBUG=SEQ_DEBUG+1
                ## SEQ_DEBUG

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
        Dprint(ws+" "+str(type(stmt)), DEBUG)
        G = generate_nodes(stmt.body, G, depth, ws, DEBUG)
        return G

    else: # Generate Data Nodes
        g = nx.DiGraph()
        g.add_node(0, name="DUMMY")
        g = generate_data_nodes(stmt, g, 0, ws, DEBUG)
        g.remove_node(0)
        return g

def generate_data_nodes(stmt, g, parent, ws, DEBUG):
    if isinstance(stmt, _stmt.BufferStore):
        g = visit_STORE(stmt, g, parent, ws, DEBUG)
    elif isinstance(stmt, _stmt.LetStmt):
        g = visit_LET(stmt, g, parent, ws, DEBUG)
    elif isinstance(stmt, tvm.ir.PrimExpr):
        g = visit_EXPR(stmt, g, parent, ws, DEBUG)
    else:
        raise NotImplementedError("Not Implemented Error: %s"%(str(type(stmt))))
    return g

def visit_STORE(stmt, g, parent, ws, DEBUG):
    node_num = len(g.nodes)
    Dprint(ws+" "+str(type(stmt))+"\t[Memory](%d)"%(node_num), DEBUG)
    Dprint(ws+" ##################### "+"@Location to be stored"+" #####################", DEBUG)
    g.add_node(node_num, name=str(stmt), type='Store')
    g.add_edge(node_num, parent)
    for idx in range(0, len(stmt.indices)):
        g = generate_data_nodes(stmt.indices[idx], g, node_num, ws+str(idx), DEBUG)
    Dprint(ws+" ##################### "+"@Value to be stored"+" #####################", DEBUG)
    g = generate_data_nodes(stmt.value, g, node_num, ws+'|', DEBUG)
    return g

def visit_LET(stmt, g, parent, ws, DEBUG):
    g = generate_data_nodes(stmt.var, g, parent, ws, DEBUG) # cse_var_1: int32
    var_num = len(g.nodes)-1
    g = generate_data_nodes(stmt.value, g, len(g.nodes)-1, ws, DEBUG) # (((i1: int32*25) + (i2: int32*5)) + i3: int32)
    # Remove edge connection with parent node before graph merge
    [g.remove_edge(node_num, parent) for node_num in g.nodes if g.has_edge(node_num, parent)]

    ## Make temporary DiGraph to merge w/ the original one
    g_ = nx.DiGraph()
    g_ = generate_nodes(stmt.body, g_, 0, ws, DEBUG) # pad_temp[cse_var_1] = placeholder[cse_var_1]

    ## Merge two graphs with new node_num
    u = nx.disjoint_union(g, g_)

    ## Merge nodes that have same name as 'var_num'
    merge_num = [num for num in u.nodes if (u.nodes[num]['name']==u.nodes[var_num]['name']) and (num is not var_num)]
    for n in merge_num:
        u = nx.contracted_nodes(u, var_num, n)
    [u.add_edge(sink, parent) for sink in u.nodes if u.out_degree(sink)==0 and not sink==0] # sink==0: KERNEL_END node
    return u

def visit_EXPR(expr, g, parent, ws, DEBUG): # tir.expr.__
    if hasattr(expr, "indices"): # BufferLoad
        node_num = len(g.nodes)
        Dprint(ws+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        g.add_node(node_num, name=str(expr), type='Load')
        g.add_edge(node_num, parent)
        for idx in range(0, len(expr.indices)):
            g = visit_EXPR(expr.indices[idx], g, node_num, ws+str(idx), DEBUG)
    elif all(hasattr(expr, attr) for attr in ["a", "b"]): # Add, Sub, Mul, Div, Mod, FloorDiv, FloorMod, Min, Max, EA, NE, LT, LE, GT, GE, And, Or
        node_num = len(g.nodes)
        Dprint(ws+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        try:
            found = re.search('<class \'tvm.tir.expr.(.+?)\'>', str(type(expr))).group(1)
        except AttributeError:
            print("%s NOT supported" %(type(expr)))
            assert 0
        g.add_node(node_num, name=str(expr), type=found)
        g.add_edge(node_num, parent)
        g = visit_EXPR(expr.a, g, node_num, ws+"L", DEBUG)
        g = visit_EXPR(expr.b, g, node_num, ws+"R", DEBUG)
    elif isinstance(expr, _expr.Cast): # Cast
        node_num = len(g.nodes)
        Dprint(ws+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        g = visit_EXPR(expr.value, g, parent, ws+"*", DEBUG)
    elif isinstance(expr, _expr.Call): # Call
        Dprint(ws+" "+str(type(expr.op))+" --> "+str(expr.op), DEBUG)
        for idx in range(0, len(expr.args)):
            g = visit_EXPR(expr.args[idx], g, parent, ws+str(idx), DEBUG)
    elif isinstance(expr, (_expr.Var)): # Var
        node_num = len(g.nodes)
        if node_num == parent: assert 0, str(expr)
        if expr.dtype == 'int32':
            Dprint(ws+" "+str(type(expr))+" --> "+str(expr), DEBUG)
            g.add_node(node_num, name=str(expr), type='Int Var')
            g.add_edge(node_num, parent)
        elif expr.dtype == 'float32':
            Dprint(ws+" "+str(type(expr))+" --> "+str(expr), DEBUG)
            g.add_node(node_num, name=str(expr), type='Float Var')
            g.add_edge(node_num, parent)
        else:
            raise NotImplementedError("Currently NOT supported variable type: %s"%(expr.dtype))
    elif isinstance(expr, (_expr.IntImm)): # IntImm
        node_num = len(g.nodes)
        Dprint(ws+" "+str(type(expr))+" --> "+str(expr), DEBUG)
        g.add_node(node_num, name=str(expr), type='Int Imm')
        g.add_edge(node_num, parent)
    elif isinstance(expr, (_expr.FloatImm)): # FloatImm
        node_num = len(g.nodes)
        Dprint(ws+" "+str(type(expr))+" --> "+str(expr), DEBUG)
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


def color_misc_nodes(g, DEBUG):
    misc_nodes = [node for node in g.nodes if g.nodes[node]['slot']=='MISC']
    while len(misc_nodes):
        for misc in misc_nodes:
            parents = list(g.predecessors(misc))
            if all([g.nodes[parent]['slot'] in ['Scalar', 'Memory', 'Vector'] for parent in parents]):
                parent_slots = []
                for parent in parents:
                    parent_slots.append(g.nodes[parent]['slot'])
                if all(e == 'Scalar' for e in parent_slots):
                    g.nodes[misc]['slot'] = 'Scalar' ## (S, S) -> S
                elif all(e in ['Memory', 'Vector'] for e in parent_slots): ## (M, M), (M, V), (V, V) -> V
                    g.nodes[misc]['slot'] = 'Vector'
                else:
                    raise RuntimeError("MISC node's parents must have same slot type: %s"%(parent_slots))
                misc_nodes.remove(misc)
                break
    return g