from typing import List
import networkx as nx
from itertools import permutations

class Op:
    def __init__(self, name: str, reads: List[List[str]], writes: List[List[str]], is_async: List[int]):
        self.name = name
        self.reads = reads
        self.writes = writes
        self.is_async = is_async

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return str(self)

MMA0 = Op('MMA0', [[None], ['Q_shared', 'K_shared', 'acc_s']], [['acc_s'], ['acc_s']], [1])
Softmax = Op('Softmax', [['scores_max'], [None], ['acc_s'], ['scores_max', 'scores_max_prev'], ['acc_s', 'scores_max'],['acc_s'],['logsum','scores_scale','scores_sum'], ['acc_s']], [['scores_max_prev'], ['scores_max'], ['scores_max'], ['scores_scale'], ['acc_s'], ['scores_sum'], ['logsum'], ['acc_s_cast']], [])
Rescale = Op('Rescale', [['acc_o', 'scores_scale']], [['acc_o']], [])
MMA1 = Op('MMA1', [['V_shared', 'acc_s_cast', 'acc_o']], [['acc_o']], [0])

graph = nx.DiGraph()
graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
nodes = [MMA0, Softmax, Rescale, MMA1]

for edge in graph.edges:
    print(edge)

max_stream = 2
def get_issue_info(graph: nx.DiGraph):
    n_nodes = len(graph.nodes)
    print("n_nodes:", n_nodes)
    def get_order():
        # order of op0 must be 0
        nodes = [i for i in range(1, n_nodes)]
        partial_all_orders =  [list(order) for order in permutations(nodes)]
        all_orders = [[0] + order for order in partial_all_orders]
        return all_orders

    def validate(order, stage) -> bool:
        for i in range(n_nodes):
            # print("i:", i)
            # print("nx.descendants(graph, i):", [(s, type(s)) for s in nx.descendants(graph, i)])
            for j in range(n_nodes):
                if (i == j):
                    continue
                if j not in nx.descendants(graph, i):
                    continue
                if stage[i] < stage[j]:
                    continue
                if stage[i] == stage[j]:
                    if order[i] > order[j]:
                        return False
                if stage[i] > stage[j]:
                    return False
        return True
        
    def get_stage(order):
        # stage of op0 must be 0
        valid_stages = []
        def gen(n, max_value):
            if n == 0:
                return [[]]
            res = gen(n - 1, max_value)
            return [item + [i] for item in res for i in range(max_value + 1)]

        partial_all_stages = gen(n_nodes - 1, max_stream)
        all_stages = [[0] + item for item in partial_all_stages]
        # print("all_stages:", len(all_stages))
        for stage in all_stages:
            if validate(order, stage):
                valid_stages.append(stage)
        # print("valid_stages:", len(valid_stages))
        return valid_stages

    ans = []
    orders = get_order()

    # print("orders:", orders)
    for order in orders:
        stages = get_stage(order)
        for stage in stages:
            ans.append((order, stage))
    return ans

def get_sync_info(graph: nx.DiGraph, issue_info):
    reads = []
    writes = []
    asyncs = []
    n_nodes = len(graph.nodes)
    def extract_stmts(graph):
        cur_id = 0
        for i in range(n_nodes):
            assert len(nodes[i].reads) == len(nodes[i].writes)
            reads.extend(nodes[i].reads)
            writes.extend(nodes[i].writes)
            if len(nodes[i].is_async) > 0:
                for async_stmt in nodes[i].is_async:
                    asyncs.append((i, async_stmt))
            cur_id += len(nodes[i].reads)
        return reads, writes, asyncs
    
    def has_intersects(l0, l1):
        for item in l0:
            if item in l1:
                return True
        return False
    
    def get_valid_pos(sync_node, mma_node, async_stmt):
        orders = issue_info[0]
        stages = issue_info[1]
        #Step 1. Check if sync before this node is possible
        mma_reads = nodes[mma_node].reads[async_stmt]
        mma_writes = nodes[mma_node].writes[async_stmt]
        if orders[sync_node] > orders[mma_node]:
            for mid_node in range(n_nodes):
                if orders[mid_node] >= orders[sync_node] \
                    or orders[mid_node] <= orders[mma_node]:
                    continue
                op_reads = [r for rs in nodes[mid_node].reads for r in rs]
                op_writes = [w for ws in nodes[mid_node].writes for w in ws]
                if stages[mid_node] == stages[mma_node] \
                    and has_intersects(mma_writes, op_reads):
                    return None
        if orders[sync_node] <= orders[mma_node]:
            # Check from mma_node to the end
            for mid_node in range(n_nodes):
                if orders[mid_node] <= orders[mma_node]:
                    continue
                op_reads = [r for rs in nodes[mid_node].reads for r in rs]
                op_writes = [w for ws in nodes[mid_node].writes for w in ws]
                if stages[mid_node] == stages[mma_node] \
                    and has_intersects(mma_writes, op_reads):
                    return None
            # Check from the start to sync node
            for mid_node in range(n_nodes):
                if orders[mid_node] >= orders[sync_node]:
                    continue
                op_reads = [r for rs in nodes[mid_node].reads for r in rs]
                op_writes = [w for ws in nodes[mid_node].writes for w in ws]
                if stages[mid_node] == stages[mma_node] + 1 \
                    and has_intersects(mma_writes, op_reads):
                    return None
                       
        # Step 2. Find the lateset possible sync position in the node 
        stmt_num = len(nodes[sync_node].reads)
        valid_pos = -1
        for i in range(stmt_num):
            stmt_reads = nodes[sync_node].reads[i]
            stmt_writes = nodes[sync_node].writes[i]
            if orders[sync_node] < orders[mma_node] \
                and stages[sync_node] == stages[mma_node] + 1 \
                and has_intersects(mma_writes, stmt_reads):
                break
            if orders[sync_node] > orders[mma_node] \
                and stages[sync_node] == stages[mma_node] \
                and has_intersects(mma_writes, stmt_reads):
                break
            valid_pos += 1

        if valid_pos == -1:
            return None
        return valid_pos
    
    async_stmt_id = 0
    # we try to put the sync as close to the end as possible
    extract_stmts(graph)
    print("reads:", reads)
    print("writes:", writes)
    print("asyncs:", asyncs)
    all_sync_pos = []
    for node, async_stmt in asyncs:
        sync_pos_list = []
        pre_stmt_num = 0
        for sync_node in range(n_nodes):
            sync_pos = get_valid_pos(sync_node, node, async_stmt)
            # print("sync_pos:", sync_pos + pre_stmt_num if sync_pos is not None else None)
            if sync_pos is not None:
                sync_pos_list.append(sync_pos + pre_stmt_num)
            pre_stmt_num += len(nodes[sync_node].reads)
        # print("sync_pos_list:", sync_pos_list)
        all_sync_pos.append(sync_pos_list)
    print("all_sync_pos:", all_sync_pos)
    return all_sync_pos

def gen_configs(graph: nx.DiGraph):
    config = []
    issue_infos = get_issue_info(graph)
    print("issue_infos:", issue_infos)
    print("issue_infos:", len(issue_infos))
    for issue_info in issue_infos:
        issue_info = ([0,1,2,3], [0,0,0,0])
        syncs = get_sync_info(graph, issue_info)
        break
        for sync in syncs:
            config.append((issue_info, sync))
    return config

pipeline_configs = gen_configs(graph)