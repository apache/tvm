'''
Extract tasks from Relay IR models and randomly sample from the mutations
of subgraphs.
'''

# import json

import tvm
from tvm import meta_schedule as ms
# from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.search_strategy import EvolutionarySearch
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.meta_schedule.task_scheduler import RoundRobin
from tvm.meta_schedule.testing.relay_workload import _load_cache
from tvm.meta_schedule.testing.utils import DummyDatabase
from tvm.meta_schedule.tune import DefaultCUDA
#from tvm.ir import save_json
from tvm.ir import load_json
from tvm.runtime import load_param_dict
from tvm.target import Target


if __name__ == '__main__':
    #mod, params, _ = get_network(name='resnet_18', input_shape=[1, 3, 224, 224],
    #                             cache_dir='/home/kchen/tvm/tmp')
    cached = _load_cache('/home/kchen/tvm/tmp', 'relay-resnet_18-1,3,224,224.json')
    mod, params_bytearray, inputs = cached
    params = load_param_dict(params_bytearray)

    all_tasks = ms.extract_task_from_relay(mod, target='cuda', params=params)
    task = all_tasks[0]
    task_name = task.task_name
    subgraph = task.dispatched[0]
    check_spatial = tvm.get_global_func('tir.schedule.IsSpatialPrimFunc')
    prim_func = subgraph[subgraph.get_global_vars()[0]]
    is_spatial = check_spatial(prim_func)
    #subgraph_str = save_json(subgraph)
    #subgraph_obj = json.loads(subgraph_str)
    #subgraph_str = json.dumps(subgraph_obj)
    #with open('/home/kchen/tvm/tmp/tir_init.json', 'w') as f:
    #    f.write(subgraph_str)

    with open('/home/kchen/tvm/tmp/tir_init.json', 'rb') as f:
        lines = f.readlines()
        tir_str = lines[0]
        tir_init = load_json(tir_str)

    num_trials_per_iter = 64
    max_trials_per_task = 400

    strategy = EvolutionarySearch(
        num_trials_per_iter=num_trials_per_iter,
        max_trials_per_task=max_trials_per_task,
    )

    # pylint: disable=protected-access
    context = TuneContext(
        mod=tir_init,
        target=Target('nvidia/geforce-rtx-3070'),
        space_generator=PostOrderApply(),
        search_strategy=strategy,
        sch_rules=DefaultCUDA._sch_rules(),
        postprocs=DefaultCUDA._postproc(),
        mutator_probs=DefaultCUDA._mutator_probs(),
        task_name=task_name,
    )
    _scheduler = RoundRobin(
        tasks=[context],
        task_weights=[1.0],
        builder=ms.builder.LocalBuilder(),
        runner=ms.runner.LocalRunner(),
        database=DummyDatabase(),
        cost_model=ms.cost_model.RandomModel(),
        measure_callbacks=[],
        max_trials=1,
    )
    context.initialize()
    context.space_generator.initialize_with_tune_context(context)
    schs = context.space_generator.generate_design_space(context.mod)
    # just containing one tir schedule

    strategy.initialize_with_tune_context(context)
    strategy.pre_tuning(schs)

    sample_init_population = tvm.get_global_func(
        'meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation'
    )
    states = sample_init_population(strategy, 2560)
    # contains a lot of tir schedule
    print(len(states))
    
    evolve_with_cost_model = tvm.get_global_func(
        'meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel'
    )
    states = evolve_with_cost_model(strategy, states, len(states))
    print(len(states))
    del _scheduler

