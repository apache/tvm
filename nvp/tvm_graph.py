import networkx as nx

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