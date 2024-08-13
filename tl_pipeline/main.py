from pipeline_transform import *
from generate_plan import * 

if __name__ == "__main__":
    _, graph = load_model("inputs.json")
    print(graph)
    print("Nodes:", graph.nodes(data=True))
    print("Edges:", graph.edges(data=True))

    topo_ordered_nodes = list(nx.topological_sort(graph))
    print("topo_ordered_nodes:", topo_ordered_nodes)

    results = []
    add_backward_edges(graph, graph.copy(), list(reversed(topo_ordered_nodes)), stream=2, cur_id=0, dst_id=0, result=results)
    plan_id = 0
    plans = []
    for i, g in enumerate(results):
        print(f"Graph {i + 1}:")
        print(g.edges(data=True))
        for p in generate(g, topo_ordered_nodes):
            p.set_graph_id(i + 1)
            plans.append(p)

    plans = list(set(plans))
    print(f"Plans: {len(plans)}")
    for i, plan in enumerate(plans):
        print(f"Plan {i + 1}:")
        print("-" * 100)
        print("Graph:", plan.graph_id)
        print(plan)
