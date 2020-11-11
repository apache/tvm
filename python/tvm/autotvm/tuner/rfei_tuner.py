from .model_based_tuner import ModelOptimizer, ModelBasedTuner
from .sa_model_optimizer import SimulatedAnnealingOptimizer
from .rf_cost_model import RFEICostModel

class RFEITuner(ModelBasedTuner):
    def __init__(
        self, 
        task, 
        plan_size=32,
        feature_type='itervar',
        loss_type='rank', 
        num_threads=None,
        optimizer='sa', 
        diversity_filter_ratio=None, 
        log_interval=50, 
        uncertainty_aware=False):
        
        cost_model = RFEICostModel(task, fea_type=feature_type)
        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval, parallel_size=plan_size*2)
        else:
            assert isinstance(optimizer, ModelOptimizer), (
                "Optimizer must be " "a supported name string" "or a ModelOptimizer object."
            )
        super(RFEITuner, self).__init__(
            task, cost_model, optimizer, plan_size, diversity_filter_ratio, uncertainty_aware
            )
        
    def tune(self, *args, **kwargs):  # pylint: disable=arguments-differ
        super(RFTuner, self).tune(*args, **kwargs)
        # manually close pool to avoid multiprocessing issues
        self.cost_model._close_pool()