# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
The search policies for TVM Auto-scheduler.

This contains the strategies to generate a schedule automatically. We provide an EmptyPolicy
which always returns an unchanged initial state, and a more advanced SketchPolicy which can
deal with various ops/subgraphs on different target devices.

Reference:
L. Zheng, C. Jia, M. Sun, Z. Wu, C. Yu, et al. "Ansor : Generating High-Performance Tensor
Programs for Deep Learning." arXiv preprint arXiv:2006.06762 (2020).
"""

import random

import tvm._ffi
from tvm.runtime import Object
from .cost_model import RandomModel
from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.SearchCallback")
class SearchCallback(Object):
    """Callback function before or after search process"""


@tvm._ffi.register_object("auto_scheduler.PreloadMeasuredStates")
class PreloadMeasuredStates(SearchCallback):
    """A SearchCallback to load measured states from the log file for a search policy.

    This can resume the state of the search policy:
        - Making sure an already measured state in former searches will never be measured again.
        - The history states can be used to speed up the search process(e.g. SketchPolicy uses
          history states as starting point to perform Evolutionary Search).

    Parameters
    ----------
    filename : str
        The name of the record file.
    """

    def __init__(self, filename="auto_scheduler_tuning.json"):
        self.__init_handle_by_constructor__(_ffi_api.PreloadMeasuredStates, filename)


@tvm._ffi.register_object("auto_scheduler.SearchPolicy")
class SearchPolicy(Object):
    """ The base class of search policies. """


@tvm._ffi.register_object("auto_scheduler.EmptyPolicy")
class EmptyPolicy(SearchPolicy):
    """This is an example empty search policy which will always generate
    the init state of ComputeDAG.

    Parameters
    ----------
    task : SearchTask
        The SearchTask for the computation declaration.
    init_search_callbacks : Optional[List[SearchCallback]]
        Callback functions called before the search process.
    """

    def __init__(self, task, init_search_callbacks=None):
        self.__init_handle_by_constructor__(_ffi_api.EmptyPolicy, task, init_search_callbacks)


@tvm._ffi.register_object("auto_scheduler.SketchPolicy")
class SketchPolicy(SearchPolicy):
    """The search policy that searches in a hierarchical search space defined by sketches.
    The policy randomly samples programs from the space defined by sketches and use evolutionary
    search to fine-tune them.

    Parameters
    ----------
    task : SearchTask
        The SearchTask for the computation declaration.
    schedule_cost_model : CostModel = RandomModel()
        The cost model to estimate the complete schedules.
    params : Optional[Dict[str, Any]]
        Parameters of the search policy.
        See `src/auto_scheduler/search_policy/sketch_search_policy.h` for the definitions.
        See `DEFAULT_PARAMS` below to find the default values.
    seed : Optional[int]
        Random seed.
    verbose : int = 1
        Verbosity level. 0 for silent, 1 to output information during schedule search.
    init_search_callbacks : Optional[List[SearchCallback]]
        Callback functions called before the search process, usually used to do extra
        initializations.
        Possible callbacks:
            - auto_scheduler.PreloadMeasuredStates
            - auto_scheduler.PreloadCustomSketchRule
            TODO(jcf94): Add these search callback implementations.
    """

    DEFAULT_PARAMS = {
        "eps_greedy": 0.05,
        "retry_search_one_round_on_empty": 10,
        "evolutionary_search_population": 2048,
        "evolutionary_search_num_iters": 10,
        "evolutionary_search_mutation_prob": 0.85,
        "evolutionary_search_use_measured_ratio": 0.2,
        "cpu_multi_level_tiling_structure": "SSRSRS",
        "gpu_multi_level_tiling_structure": "SSSRRSRS",
        # Notice: the default thread bind policy of GPU assumes the tiling structure to have at
        # least 3 spatial tiling levels in outermost
        "max_innermost_split_factor": 16,
        "max_vectorize_size": 16,
        "disable_change_compute_location": 0,
    }

    def __init__(
        self,
        task,
        schedule_cost_model=RandomModel(),
        params=None,
        seed=None,
        verbose=1,
        init_search_callbacks=None,
    ):
        if params is None:
            params = SketchPolicy.DEFAULT_PARAMS
        else:
            for key, value in SketchPolicy.DEFAULT_PARAMS.items():
                if key not in params:
                    params[key] = value

        self.__init_handle_by_constructor__(
            _ffi_api.SketchPolicy,
            task,
            schedule_cost_model,
            params,
            seed or random.randint(1, 1 << 30),
            verbose,
            init_search_callbacks,
        )

    def generate_sketches(self, print_for_debug=False):
        """Generate the sketches.
        This python interface is mainly used for debugging and testing.
        The actual search is all done in c++.

        Parameters
        ----------
        print_for_debug : bool = False
            Whether print out the sketches for debug.

        Returns
        -------
        sketches : List[State]
            The generated sketches of this search task.
        """
        sketches = _ffi_api.SketchPolicyGenerateSketches(self)
        if print_for_debug:
            for i, s in enumerate(sketches):
                print("=" * 20 + " %d " % i + "=" * 20)
                print(s)
        return sketches

    def sample_initial_population(self, pop_size):
        """Sample initial population.
        This python interface is mainly used for debugging and testing.
        The actual search is all done in c++.

        Parameters
        ----------
        pop_size : int
            The size of sampled population

        Returns
        -------
        states: List[State]
            The sampled states
        """
        states = _ffi_api.SketchPolicySampleInitialPopulation(self, pop_size)
        return states

    def evolutionary_search(self, init_populuations, out_size):
        """Evolutionary search.
        This python interface is mainly used for debugging and testing.
        The actual search is all doen in c++.
        Parameters
        ----------
        init_populations: List[State]
            The initial population states
        out_size : int
            The size of generated states
        Returns
        -------
        states: List[State]
            The generated states
        """
        states = _ffi_api.SketchPolicyEvolutionarySearch(self, init_populuations, out_size)
        return states
