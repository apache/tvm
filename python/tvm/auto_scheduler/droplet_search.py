import os
from tvm.auto_scheduler.space import Space
from tvm.auto_scheduler.search_task import SearchTask
from tvm.autotvm.tuner.droplet_tuner import DropletTuner

from .utils import *


class Droplet(DropletTuner):
    """Tuner with droplet algorithm in Ansor.

    Parameters
    ----------
    json_file: str
        json format file
    target:
        hardware target
    log: str
        path to save json file
    trials: int
        number of samples, the default is 100
    pvalue: float
        statistical value to confidence level, the default is 0.05
    """

    def __init__(self, json_file, target, log, trials=100, pvalue=0.05) -> None:
        workload_key = json_file["i"][0][0]
        self.task = SearchTask(workload_key=workload_key, target=target)
        super(DropletTuner, self).__init__(self.task)
        self.space = Space(json_file, self.task)
        self.final_log = write_file([json_file], log)
        self.log = write_file([json_file])
        self.trials, self.pvalue = trials, pvalue
        self.next = [(0, [0] * len(self.space.dims))]
        best_avg, _, _ = get_time(self.log)
        self.best_choice = [0, [0] * len(self.space.dims), best_avg]
        self.count, self.execution, self.found_best_pos = 1, 1, True
        self.visited, self.batch = set([0]), max(os.cpu_count(), 16)
        self.total_execution = 1
        if len(self.space.dims) > 0:
            self.total_execution = max(self.space.dims)
        self.dims, self.step = self.space.dims, 1

    def next_batch(self, batch_size):
        i, json_file_list = 0, []
        for i in range(i < len(self.next)):
            if batch_size > 0 and self.count >= self.trials:
                break
            json_file_list.append(self.space.apply_opt(self.next[i][1]))
            i, self.count = i + 1, self.count + 1
        log = write_file(json_file_list)
        return self.space.run(log, self.final_log)

    def has_next(self):
        return len(self.next) > 0 and self.found_best_pos

    def tune(self):
        self.speculation()
        while self.has_next():
            ins, res = self.next_batch(self.batch)
            self.update(ins, res)
