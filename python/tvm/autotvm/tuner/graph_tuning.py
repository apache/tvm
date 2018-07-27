"""Utilities for tuning a whole graph (a set of tasks)"""

import os
from . import callback, XGBTuner, GATuner, RandomTuner, GridSearchTuner
from .. import record, task

def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=500,
               early_stopping=200,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    """
    Tune a set of tasks

    Parameters
    ----------
    tasks: Array of Task
        A list of tasks to tune
    rpc_device_key: str
        The key of devices in rpc tracker
    tuner: str
        The type of tuner.
        If is 'xgb', use :any:`XGBTuner`.
        If is 'ga', use :any:`GATuner`.
        If is 'random', use :any:`RandomTuner`.
        If is 'gridsearch', use :any:`GridSearchTuner`.
    n_trial: int
        The maximum number of trials for a workload
    early_stopping: int
        The early stopping metric. The tuner will stop when it cannot find better
        config after `early_stopping` trials
    log_filename: str
        The filename of output log file to store best configs
    use_transfer_learning: bool
        Whether reuse history tuning log to accelerate tuning
    try_winograd: bool
        Whether try to use winograd template
    """
    if try_winograd:
        for i in range(len(tasks)):  # pylint:disable=consider-using-enumerate
            try:  # try winograd template
                tsk = task.create(tasks[i].name, tasks[i].args,
                                  tasks[i].target, tasks[i].target_host,
                                  'winograd')
                tasks.append(tsk)
            except Exception:  # pylint:disable=broad-except
                pass

    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank', verbose=0)
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           callback.progress_bar(n_trial, prefix=prefix),
                           callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
