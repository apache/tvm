import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## layout
# layouts = {'spring': nx.spring_layout(G), 
#            'spectral':nx.spectral_layout(G), 
#            'shell':nx.shell_layout(G), 
#            'circular':nx.circular_layout(G),
#            'kamada_kawai':nx.kamada_kawai_layout(G), 
#            'random':nx.random_layout(G)
#           }

def save_graph_viz(g, viz, dir='graph.png'):
    print('########## PLOTTING (%s) ##########'%(dir))
    empty_idxes = [idx for idx in g.nodes if not g.nodes[idx]] # remove empty node
    g.remove_nodes_from(empty_idxes)
    # print('Nodes List: %s'%(g.nodes))
    # [print(["%s"%(idx), g.nodes[idx]]) for idx in g.nodes]

    f = plt.figure(figsize=(12, 9))
    color_map = []
    for node in g.nodes:
        if not 'slot' in g.nodes[node]: color_map.append('grey')
        elif g.nodes[node]['slot'] == 'Scalar': color_map.append('red')
        elif g.nodes[node]['slot'] == 'Memory': color_map.append('green')
        elif g.nodes[node]['slot'] == 'Vector': color_map.append('cyan')
        elif g.nodes[node]['slot'] == 'Control': color_map.append('orange')
        else:
            raise NotImplementedError("Currently Not Supported Slot Type: %s"%(g.nodes[node]))

    if viz == 'type':
        labels = nx.get_node_attributes(g, 'type')
    elif viz == 'typeNum':
        labels = nx.get_node_attributes(g, 'type')
        for key in labels.keys():
            labels[key] = str(labels[key]) + str('(%d)'%(key))
    elif viz == 'name':
        labels = nx.get_node_attributes(g, 'name')
    elif viz == 'num':
        labels = None
    else:
        raise NotImplementedError('Currently NOT implemented: %s' %(viz))
    # layout = nx.spring_layout(g)
    layout = nx.kamada_kawai_layout(g)
    # layout = nx.shell_layout(g)
    nx.draw(g, pos=layout, node_color=color_map, labels=labels, with_labels=True, font_size=12)
    edge_labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos=layout, edge_labels=edge_labels)
    f.savefig(dir)
