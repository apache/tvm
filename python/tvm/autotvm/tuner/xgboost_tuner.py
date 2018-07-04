from .model_based_tuner import ModelBasedTuner, ModelOptimizer
from .xgboost_cost_model import XGBoostCostModel
from .sa_model_optimizer import SimulatedAnnealingOptimizer

class XGBTuner(ModelBasedTuner):
    def __init__(self, task, plan_size,
                 feature_type='itervar', loss_type='rank', num_threads=None,
                 optimizer='sa'):
        cost_model = XGBoostCostModel(task,
                                      feature_type=feature_type,
                                      loss_type=loss_type,
                                      num_threads=num_threads)
        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."

        super(XGBTuner, self).__init__(task, cost_model, optimizer, plan_size)
